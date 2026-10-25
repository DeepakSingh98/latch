"""Opt-in matrix provider for the frozen compiler's reference (generic dependency) schedule.

``MatrixProvider`` is consulted by ``ops/fusion_reference.py`` for every ``mm``/``bmm`` when passed as
``matrix_provider=``. It converts the captured operand views
into a ``MatrixProblem``, applies one selection policy under the program's declared resources, and returns the
selected body's emitted code with its headers, shared-memory requirement and decision record. Returning None keeps
the frozen path for that product (FP32 products, refused problems, or bodies the frozen path already emits).

The composed compiler owns this provider per lowering under its existing lock.
No process-global installation or donor-checkout dependency is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import re

from .candidates import select
from ..explain.collect import note as _explain_note  # LATCH-EXPLAIN: no-op unless a collector is active
from .problem import MatrixProblem, Operand
from .requirements import ProgramResources

FROZEN_BODIES = ('popsc_bf16', 'popsc_bf16_tail', 'mixed_generic', 'fp32_pops2')
# Step 16: an explicit per-problem body choice installed by ops.matrix_tuning for one lowering. ``None``
# (the default) leaves every selection exactly as before.
_CHOICES = ContextVar('latch_matrix_choices', default=None)


def problem_key(problem):
    """Stable string form of ``MatrixProblem.key()``: extents, strides, offsets, dtypes and result mode."""
    return json.dumps(problem.key(), separators=(',', ':'))


@contextmanager
def matrix_choices(choices, *, log=None, policy=None):
    """Install ``{problem_key: body | 'native'}`` for the composed provider of lowerings in this context.

    Every product is reported to ``log`` (inventory), including products the native path keeps (P5C).
    A choice is emitted only when its body is admitted by the existing structural/resource check; otherwise the
    existing legal selection is kept and the rejection is recorded. ``policy`` (step 13, optional) names the
    default-selection policy of ``candidates.select`` used for products without a choice; ``None`` keeps the
    composed planner's default.
    """
    token = _CHOICES.set({'choices': dict(choices or {}), 'log': log if log is not None else [], 'policy': policy})
    try:
        yield
    finally:
        _CHOICES.reset(token)


_POLICY_OVERRIDE = ContextVar('latch_matrix_default_policy', default=None)


@contextmanager
def default_policy(name):
    """Step 13: default-selection policy for composed lowerings in this context (explicit scope policies win)."""
    token = _POLICY_OVERRIDE.set(name)
    try:
        yield
    finally:
        _POLICY_OVERRIDE.reset(token)


def compose_policy():
    """The provider policy of a composed lowering: explicit scope policy, else override, else the default."""
    from .candidates import COMPOSE_DEFAULT_POLICY
    scope = _CHOICES.get()
    if scope is not None and scope.get('policy') is not None:
        return scope['policy']
    return _POLICY_OVERRIDE.get() or COMPOSE_DEFAULT_POLICY


def current_choices():
    """Provider keyword arguments for the active ``matrix_choices`` scope (empty outside one)."""
    scope = _CHOICES.get()
    if scope is None:
        return {}
    out = {'choices': scope['choices'], 'inventory': scope['log']}
    if scope.get('policy') is not None:
        out['policy'] = scope['policy']
    return out


def problem_from_layouts(la, lb, shape, dtypes, *, out='C', out_mode='store'):
    """MatrixProblem from frozen-compiler TensorLayouts (rank 2 or 3)."""
    if len(la.shape) not in (2, 3) or len(lb.shape) != len(la.shape):
        return None
    m, k = la.shape[-2:]
    kb, n = lb.shape[-2:]
    batch = la.shape[0] if len(la.shape) == 3 else 1
    if k != kb or tuple(shape) != ((batch, m, n) if batch > 1 or len(la.shape) == 3 else (m, n)):
        return None
    if len(la.shape) == 3 and lb.shape[0] != batch:
        return None
    da, db, dc = dtypes
    if da != db:
        return None
    oa = Operand(la.root, la.strides[-2], la.strides[-1], la.strides[0] if len(la.shape) == 3 else 0, la.offset)
    ob = Operand(lb.root, lb.strides[-2], lb.strides[-1], lb.strides[0] if len(lb.shape) == 3 else 0, lb.offset)
    if batch == 1 and len(la.shape) == 3:
        return None          # rank-3 batch-1 views keep the frozen path (shape contract differs)
    return MatrixProblem(m, n, k, oa, ob, out=out, batch=batch, dtype=da, out_dtype=dc, out_mode=out_mode)


class MatrixProvider:
    def __init__(self, policy='donor', resources=None, *, take_frozen=False, only=None, force=None, log=None,
                 choices=None, inventory=None, serial_bf16=None, matrix_accumulation_contract=None,
                 matrix_splitk_contract=None):
        self.policy = policy
        # Step 16 (opt-in): per-problem choices; the provider is then also consulted for natively kept products.
        self.choices = choices
        self.inventory = inventory
        self.serial_bf16 = serial_bf16 or {}
        self.serial_products = self.serial_bf16.get('products', {})
        from ..matrix_accumulation import validate
        self.matrix_accumulation_contract = validate(matrix_accumulation_contract)
        from ..matrix_splitk import validate as validate_splitk
        self.matrix_splitk_contract = validate_splitk(matrix_splitk_contract)
        self.consult_native = (choices is not None or bool(self.serial_products)
                               or self.matrix_accumulation_contract is not None
                               or self.matrix_splitk_contract is not None)
        self.force = force                    # diagnostic/component use: emit this body when it is admitted
        self.resources = resources or ProgramResources()
        self.take_frozen = take_frozen        # True: also emit bodies the frozen path would choose (identity tests)
        self.only = set(only) if only else None
        self.decisions = [] if log is None else log
        self.emitted = []
        self.inventory_start = len(inventory) if inventory is not None else 0

    # ---- called by fusion_reference -------------------------------------------------------------------------
    def product(self, *, a, b, name, out, la, lb, shape, dtypes, ident, target_arch, native=None, out_pitch=None,
                out_mode='store'):
        problem = problem_from_layouts(la, lb, shape, dtypes, out=name, out_mode=out_mode)
        serial = self.serial_products.get(name)
        from .. import matrix_accumulation, matrix_splitk
        explicit_splitk = (self.matrix_splitk_contract is not None
                           and not serial and matrix_splitk.applies(problem))
        explicit_accumulation = (self.matrix_accumulation_contract is not None
                                 and not serial and not explicit_splitk and matrix_accumulation.applies(problem))
        if serial and (problem is None or target_arch != 'sm_90'):
            raise ValueError(f'{name}: serial BF16 contract requires an admitted sm_90 matrix')
        if (explicit_accumulation or explicit_splitk) and target_arch != 'sm_90':
            raise ValueError(f'{name}: matrix accumulation contract requires an admitted sm_90 matrix')
        if problem is None:
            return None
        if target_arch != 'sm_90':
            return None
        if serial:
            from ..serial_bf16 import MMA_K8_CONTRACT, MMA_K8_RESIDUE_CONTRACT
            if serial.get('contract') in (MMA_K8_CONTRACT, MMA_K8_RESIDUE_CONTRACT):
                from .mma_k8 import emit
                return emit(self, problem, serial, a=a, b=b, name=name, out=out,
                            ident=ident, out_pitch=out_pitch)
        if explicit_splitk:
            if self.force is not None or (self.choices and problem_key(problem) in self.choices):
                raise ValueError(f'{name}: matrix split-K contract conflicts with a per-product body choice')
            from .splitk import emit
            return emit(self, problem, a=a, b=b, name=name, out=out, ident=ident, out_pitch=out_pitch)
        chosen, decision = select(problem, self.resources, self.policy)
        # LATCH-EXPLAIN: the manifest keeps a compact record; the full candidate table is otherwise discarded.
        _explain_note("matrix_provider_decision", output=name, native=native, decision=decision)
        step13_body = self._step13_body(problem, chosen, decision, native)
        keep_native = False
        if self.choices is not None:
            chosen, keep_native = self._choose(problem, name, chosen, decision, native)
            if keep_native and native == 'p5c' and not serial and not explicit_accumulation:
                return None              # the unchanged path never consults the provider for P5C products
        elif native == 'p5c' and not serial and not explicit_accumulation:
            return None                  # serial bindings do not alter other native products
        if (self.force is None and not serial and not explicit_accumulation
                and not (self.choices and problem_key(problem) in self.choices)):
            # LATCH-TINYMOE-MATRIX (opt-in ops.matrix_schedule): a short single-wave contraction takes the
            # equivalent cp.async wgmma body (same bitwise class); see matrix_tma_split.short_contraction.
            from ..matrix_tma_split import short_contraction
            chosen = short_contraction(problem, chosen, self.resources, decision)
        if self.force is not None:
            from .candidates import candidates as all_candidates
            forced = [c for c in all_candidates(problem, self.resources)[0] if c.body == self.force and c.admitted]
            chosen = forced[0] if forced else None
            decision['rule'] = f'forced body {self.force}' + ('' if forced else ' (not admitted)')
        if serial:
            from .candidates import candidates as all_candidates
            from ..serial_bf16 import CONTRACT
            if (serial.get('contract') != CONTRACT or serial.get('a') != a or serial.get('b') != b
                    or serial.get('shape') != [problem.m, problem.n, problem.k]
                    or dtypes != ('bfloat16', 'bfloat16', 'bfloat16')
                    or type(serial.get('partition_k')) is not int or serial['partition_k'] <= 0
                    or serial['partition_k'] % 16 or serial['partition_k'] >= problem.k):
                raise ValueError(f'{name}: graph-bound serial BF16 matrix contract changed')
            admitted = [c for c in all_candidates(problem, self.resources)[0]
                        if c.body == 'pops4b_tma' and c.admitted]
            if not admitted:
                raise ValueError(f'{name}: serial BF16 contract has no admitted native TMA body')
            chosen, keep_native = admitted[0], False
            decision['rule'] = 'explicit graph-bound serial BF16 partition epilogue'
        if explicit_accumulation:
            from .bodies import BY_NAME
            from .candidates import candidates as all_candidates
            if self.force is not None or (self.choices and problem_key(problem) in self.choices):
                raise ValueError(f'{name}: matrix accumulation contract conflicts with a per-product body choice')
            if out_pitch is not None and out_pitch != problem.n:
                raise ValueError(f'{name}: matrix accumulation contract requires an unpadded result pitch')
            matches, refused = all_candidates(problem, self.resources,
                                               bodies=tuple(BY_NAME[body] for body in matrix_accumulation.bodies()))
            # Prefer the first body of matrix_accumulation.bodies() that is admitted (identical bits, probes).
            admitted = [candidate for body in matrix_accumulation.bodies() for candidate in matches
                        if candidate.admitted and candidate.body == body]
            if not admitted:
                reasons = {candidate.body: candidate.reasons for candidate in matches}
                reasons.update(refused)
                raise ValueError(f'{name}: matrix accumulation contract body refused: {reasons}')
            chosen, keep_native, step13_body = admitted[0], False, None
            decision['rule'] = ('explicit long-contraction BF16 k16 arithmetic' +
                                ('' if chosen.body == matrix_accumulation.BODY else ' on the equivalent warpgroup body'))
        compact = {'output': name, 'provider': 'latch_matrix', 'policy': self.policy,
                   'selected': chosen.body if chosen else None, 'rule': decision['rule'],
                   'admitted': [c['body'] for c in decision['candidates'] if c['admitted']],
                   'refused': {c['body']: c['reasons'] for c in decision['candidates'] if not c['admitted']}}
        if keep_native:
            compact.update(selected=None, rule='step16 tuned choice: keep the native path')
        compact['problem_key'] = problem_key(problem)
        if step13_body is not None and not (self.choices and compact['problem_key'] in self.choices):
            compact['step13_body'] = step13_body
        self.decisions.append(compact)
        if chosen is None or keep_native:
            return None
        if not getattr(chosen, 'tuned_choice', False) and not explicit_accumulation:
            # LATCH-TRAIN-MATMUL: a P5C plan selected by the opt-in matrix_bodies 'p5c' rule is emitted by this
            # provider even where the composed planner's body set (PROVIDER_BODIES) omits P5C.
            selected_p5c = chosen.body == 'p5c' and chosen.plan.get('matrix_bodies') == 'p5c'
            if self.only is not None and chosen.body not in self.only and not selected_p5c:
                return None
            if chosen.body in FROZEN_BODIES and not self.take_frozen and self.force is None:
                return None
        # Rotation token: the reference schedule substitutes /*B_VBASE*/ by the phase's matrix tile offset.
        # Step 02: a padded result pitch is honored only by the masked TMA body (16-byte stores into the padding).
        pitch = out_pitch if out_pitch is not None and chosen.body == 'pops4b_tma' else None
        if serial:
            from .bodies import BY_NAME
            emission = BY_NAME['pops4b_tma'].emit(problem, chosen.plan, ident=ident, a=a, b=b, out=out,
                rot='/*B_VBASE*/', ldc=pitch, serial_bf16_k=serial['partition_k'])
        else:
            emission = (chosen.emit(ident=ident, a=a, b=b, out=out, rot='/*B_VBASE*/', ldc=pitch) if pitch
                        else chosen.emit(ident=ident, a=a, b=b, out=out, rot='/*B_VBASE*/'))
        req = chosen.requirements
        extra = {'tiles': emission.tiles, 'k': problem.k, 'shared_bytes': int(req.dynamic_shared_bytes), 'threads': 256,
                 'engine': f'latch_matrix:{chosen.body}', 'matrix_shape': [problem.m, problem.n, problem.k],
                 'batch': problem.batch, 'matrix_plan': {k: v for k, v in chosen.plan.items()
                                                         if isinstance(v, (int, float, str, bool))},
                 'arithmetic': f'{req.precision.instruction}; {req.precision.order}; {req.precision.rounding}',
                 'latch_matrix': {'body': chosen.body, 'bitwise_class': req.precision.bitwise_class,
                                  'arch': req.arch, 'entry_resident': req.sync.entry_resident,
                                  'host_setup': list(req.host_setup)}}
        compact['plan'] = extra['matrix_plan']
        if explicit_accumulation:
            extra[matrix_accumulation.KEY] = matrix_accumulation.record(problem, chosen.body)
            compact['arithmetic_contract'] = matrix_accumulation.record(problem, chosen.body)
        if serial:
            extra['serial_bf16_contract'] = dict(serial)
            extra['arithmetic'] = ('WGMMA FP32 ascending k16 partials from zero; '
                                   f'BF16 RN(partial+previous BF16) every {serial["partition_k"]} K')
            extra['latch_matrix']['bitwise_class'] = serial['contract']
            compact['arithmetic_contract'] = dict(serial)
        self.emitted.append(name)
        # Bind literal TMA roots before the common planner considers state aliases.
        # This is adapter metadata, not a claim that the body supports generated
        # matrix epilogues or matrix co-scheduling.
        extra['matrix_spec'] = {
            'engine': 'latch_matrix:' + chosen.body,
            'a': a, 'b': b, 'out': out, 'm': problem.m, 'n': problem.n,
            'k': problem.k, 'batch': problem.batch,
            'roots': [(la.root, int(la.offset)), (lb.root, int(lb.offset))],
        }
        if (chosen.body == 'pops4b_wg' and problem.batch == 1 and problem.out_mode == 'store' and pitch is None
                and getattr(chosen, 'short_contraction', False)):
            # LATCH-TINYMOE-MATRIX: the cp.async body chosen for a short contraction keeps joint tile planning
            # (group_rules.joint_tma_plans) through the same plain-data spec/plan record.
            from .bodies import BY_NAME
            extra['wg_respec'] = {'spec': BY_NAME['pops4b_wg']._spec(problem, a, b, out),
                                  'plan': dict(chosen.plan), 'rot': '/*B_VBASE*/'}
        if chosen.body == 'pops4b_tma' and problem.batch == 1 and problem.out_mode == 'store':
            # Release step 14: the exact body spec and plan, so a later same-index epilogue pass can re-emit this
            # product with a generated staged epilogue (matrix_adapters.bodies.respec_tma). Plain data only.
            from .bodies import BY_NAME
            extra['tma_respec'] = {'spec': BY_NAME['pops4b_tma']._spec(problem, a, b, out),
                                   'plan': dict(chosen.plan), 'rot': '/*B_VBASE*/'}
            if pitch is not None:
                # Joint tile planning can re-emit this body later. Preserve the
                # step-02 physical result pitch in that reusable specification.
                extra['tma_respec']['spec']['ldc'] = pitch
            if serial:
                extra['tma_respec']['spec']['serial_bf16_k'] = serial['partition_k']
        elif chosen.body == 'pops4b_tma' and problem.batch > 1:
            from .. import matrix_widen
            from .bodies import BY_NAME
            stacked_spec = BY_NAME['pops4b_tma']._spec(problem, a, b, out)
            extra['batch_stacked_tma'] = {
                'source_commit': '8292d176b9dbe224bd9f73804f368d68d81e22e5',
                'batch': problem.batch, 'rows_per_batch': problem.m,
                'b_map_step': problem.n if problem.bkc else problem.k,
                'b_k_contiguous': problem.bkc, 'tile_m': chosen.plan['bm'], 'tile_n': chosen.plan['bn'],
                'arithmetic': 'unchanged ascending WGMMA k16 chain and one BF16 result rounding',
            }
            if matrix_widen.BATCH_TMA_WIDEN:
                # Separate from batch-one staged epilogues and joint tile passes.
                extra['batch_widen_respec'] = {'spec': stacked_spec, 'plan': dict(chosen.plan),
                                               'rot': '/*B_VBASE*/'}
            if problem.out_mode == 'store':
                # LATCH-MOE-EMA-100: batch-stacked products may also take same-index staged epilogues
                # (group_rules.fuse_tma_epilogues, BATCHED_TMA_EPILOGUES); separate key, so joint tile passes keep
                # treating them as before.
                extra['tma_respec_batched'] = {'spec': stacked_spec, 'plan': dict(chosen.plan),
                                               'rot': '/*B_VBASE*/'}
        contract = {
            'work_items': emission.tiles,
            'execution_grid_scope': 'physical_cooperative_grid',
            'requires_full_grid': req.sync.requires_full_grid,
            'internal_mutations': [], 'counter_effects': [], 'private_scratch': [],
        }
        if pitch:
            compact['out_pitch'] = pitch
        return {'code': emission.code, 'headers': list(emission.headers), 'helpers': list(emission.helpers),
                'extra': extra, 'decision': compact, 'contract': contract, 'out_pitch': pitch}

    def tma_emission(self, **kwargs):
        """LATCH-TINYMOE-MATRIX: the provider's admitted pops4b TMA record for a product the native path keeps
        (structural split rules re-emit it as K slices), or None. Uses the provider's own admission and plan."""
        force, self.force = self.force, 'pops4b_tma'
        marks = (len(self.decisions), len(self.emitted))
        try:
            provided = self.product(**kwargs)
        finally:
            self.force = force
        if provided is None or not provided['extra'].get('tma_respec'):
            self.discard(marks)
            return None
        provided['marks'] = marks
        return provided

    def discard(self, marks):
        """Forget the records of an emission the caller did not use (the native path keeps the product)."""
        del self.decisions[marks[0]:]
        del self.emitted[marks[1]:]

    def _step13_body(self, problem, chosen, decision, native):
        """The body the step-13 common-path rule would choose instead of this default (None: no change)."""
        from .candidates import STEP13_MIN_FLOPS, STEP13_REPLACED
        if (self.policy == 'step13' or chosen is None or native == 'p5c' or problem.batch != 1
                or chosen.body not in STEP13_REPLACED or problem.flops < STEP13_MIN_FLOPS):
            return None
        admitted = {c['body'] for c in decision['candidates'] if c['admitted']}
        return next((b for b in ('pops4b_tma', 'pops4b_wg') if b in admitted), None)

    def _default_body(self, chosen, native):
        """The body the unchanged provider would emit, or 'native' when the existing native path keeps it."""
        if native == 'p5c' or chosen is None:
            return 'native'
        if self.only is not None and chosen.body not in self.only:
            return 'native'
        if chosen.body in FROZEN_BODIES and not self.take_frozen:
            return 'native'
        return chosen.body

    def _choose(self, problem, name, chosen, decision, native):
        """Apply an installed choice. Returns (candidate, keep_native); the candidate is the tuned or default one."""
        from .candidates import candidates as all_candidates
        key = problem_key(problem)
        default = self._default_body(chosen, native)
        admitted = [c for c in all_candidates(problem, self.resources)[0] if c.admitted]
        want = self.choices.get(key)
        row = {'output': name, 'key': key, 'native': native or 'fallback', 'default': default,
               'admitted': [c.body for c in admitted], 'flops': problem.flops, 'batch': problem.batch,
               'classes': {c.body: c.requirements.precision.bitwise_class for c in admitted},
               'requested': want}
        selected = default
        if want is not None and want != default:
            if want == 'native':
                selected = 'native'
            else:
                hit = next((c for c in admitted if c.body == want), None)
                if hit is None:
                    row['rejected'] = 'requested body is not admitted by the structural/resource check'
                else:
                    hit.donor_rule = 'step16 tuned choice (existing body, resource-checked)'
                    hit.tuned_choice = True
                    decision['rule'] = hit.donor_rule
                    chosen, selected = hit, want
        row['selected'] = selected
        if self.inventory is not None:
            self.inventory.append(row)
        if selected == 'native':
            # Default-native products follow the unchanged code path exactly.
            return (chosen, False) if default == 'native' and native != 'p5c' else (None, True)
        return chosen, False

    def finalize(self, program):
        """Bind P5C_TMAP / POPS4B_TMAP placeholders of provided products to map-table entries."""
        missing = set(self.serial_products) - set(self.emitted)
        if missing:
            raise ValueError(f'serial BF16 arithmetic bindings were not emitted: {sorted(missing)}')
        if not self.emitted:
            return program
        from megakernel_lab.operation_compiler.lower import GeneratedProgram
        source = program.source
        manifest = dict(program.manifest)
        manifest['latch_matrix'] = {'policy': self.policy, 'resources': self.resources.to_dict(),
                                    'decisions': list(self.decisions)}
        if self.serial_products:
            manifest['serial_bf16_readout'] = self.serial_bf16
        if 'P5C_TMAP(' not in source and 'POPS4B_TMAP(' not in source:
            return GeneratedProgram(source, manifest, program.kernel_body, program.helpers, program.functions)
        inputs = manifest['inputs']
        maps, index = list(manifest.get('tensor_maps') or ()), {}

        def bind(kind):
            def replace(match):
                key = (kind, match.group(1))
                if key not in index:
                    root, offset, d0, d1, stride, b0, b1, *rest = match.group(1).split(',')
                    layout = manifest.get('layouts', {}).get(root, {})
                    storage_root = layout.get('root', root) if isinstance(layout, dict) else root
                    element_offset = int(offset) + (int(layout.get('offset', 0)) if isinstance(layout, dict) else 0)
                    if storage_root in inputs:
                        where = {'storage': 'input', 'index': inputs.index(storage_root)}
                    elif storage_root in manifest['offsets']:
                        where = {'storage': 'arena', 'word_offset': int(manifest['offsets'][storage_root])}
                    else:
                        raise ValueError(f'tensor-map operand root {root} has no storage')
                    index[key] = len(maps)
                    maps.append({'root': storage_root, **where, 'element_offset': element_offset,
                                 'dims': [int(d0), int(d1)], 'stride_bytes': int(stride), 'box': [int(b0), int(b1)],
                                 'dtype': 'bfloat16', 'swizzle': int(rest[0]) if rest else 128, 'oob_fill': 'zero'})
                return f'(reinterpret_cast<const char*>(args[{len(inputs)}]) + {128 * index[key]})'
            return replace

        source = re.sub(r'P5C_TMAP\(([^()]*)\)', bind('p5c'), source)
        source = re.sub(r'POPS4B_TMAP\(([^()]*)\)', bind('p4b'), source)
        # ptxas serializes every wgmma of a kernel that reaches __assertfail (C7510): generated bounds checks trap.
        from megakernel_lab.operation_compiler import p5c_matrix
        entry = 'extern "C" __global__'
        if source.count(entry) == 1 and 'P-OPS5-C: generated index bounds checks' not in source:
            source = source.replace(entry, p5c_matrix.TRAP_ASSERT + entry, 1)
            manifest['index_checks'] = 'trap (assert overridden for wgmma programs)'
        manifest['tensor_maps'] = maps
        manifest['source_hash'] = hashlib.sha256(
            (source + json.dumps(manifest['header_hashes'], sort_keys=True)).encode()).hexdigest()
        return GeneratedProgram(source, manifest, program.kernel_body, program.helpers, program.functions)
