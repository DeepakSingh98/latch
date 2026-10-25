"""Explicit noinline pure runs within ordinary mixed publication intervals.

Only complete phase sequences with already-local tensor bindings move. Every
surrounding CTA barrier, resource body, caller binding and publication stays
literal. The previously qualified whole-interval mode is a separate predecessor.
"""
from collections import Counter
import hashlib
import json
import re

from . import elementwise_codegen as ec
from . import phase_pointer_codegen as pp
from . import mixed_phase_pointer_codegen as mixed
from . import resource_phase_pointer_codegen as resource
from . import pure_interval_codegen as pi

Refusal = pp.Refusal
KEY = 'pure_run_codegen'
PREFIX = 'latch_prun_body'
MIN_PHASES = 32
PARAMETERS = 'float** args, float* arena, unsigned lsr_tx, unsigned lsr_bx, unsigned lsr_nt, unsigned lsr_nb'
ACTUALS = 'args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb'


def _admit(phase, body, metadata, prior, helpers):
    if metadata.get('kind') != 'elementwise' or any(metadata.get(k) for k in
            ('counter_effects', 'private_scratch', 'requires_full_grid', 'shared_bytes')):
        raise Refusal('non-pure phase or cross-phase resources')
    if phase not in prior: raise Refusal('phase lacks prior local-pointer proof')
    clean = resource._without_prior_prefix(phase, body, prior)
    prefix = body[:len(body)-len(clean)]
    declared = {}
    for line in prefix.splitlines(keepends=True):
        if pp.POINTER.fullmatch(line):
            name, ty = pp._pointer(line)
            declared[name] = ty
    if not declared: raise Refusal('missing local pointer declarations')
    if 'out' in set(re.findall(r'\b[A-Za-z_]\w*\b', prefix)):
        raise Refusal('unbound output pointer table')
    if not pp._pure_body(phase, clean, declared, helpers) <= declared.keys():
        raise Refusal('body references an enclosing tensor')
    return len(declared)


def fold(program):
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if KEY in manifest or PREFIX in source: raise Refusal('already outlined or reserved helper name')
    if not {pi.KEY, resource.KEY, mixed.KEY} <= manifest.keys() or pp.KEY in manifest:
        raise Refusal('requires qualified whole-interval/resource/mixed predecessors')
    entries = list(ec.KERNEL.finditer(source))
    if (len(entries) != 1 or manifest.get('execution_selection', {}).get('selected') != 'common_static'
            or manifest.get('barrier_schedule') != 'dependency'
            or not {'lptr', 'lsreg'} <= set(manifest.get('group_rules', {}).get('local_pointers', ()))
            or any(manifest.get(k) for k in ('region_body', 'stages', 'dynamic_batch', 'output_placement'))):
        raise Refusal('requires ordinary standalone common Runtime')
    entry = entries[0]
    if not source[entry.end():].startswith('float** args, float* arena, unsigned long long* ticks) {\n'):
        raise Refusal('plain kernel ABI differs')
    if re.search(r'\b(?:LATCH_CTRACE|POPS4B_TRACE|latch_cargs)\b', source):
        raise Refusal('trace or constant-binding entry mode')
    definitions = re.findall(r'(?m)^[^\n]*\blatch_opq\([^;\n]*\) \{[^\n]*', source)
    if definitions != [pp.OPAQUE_HELPER] or re.search(r'(?m)^\s*#\s*define\s+latch_opq\b', source):
        raise Refusal('ordinary opaque helper differs')
    pairs = pp._braces(source)
    opening = source.index('{', entry.end()); closing = pairs[opening]
    entry_bytes = len(source[opening+1:closing].encode())
    all_bodies = list(ec._phase_bodies(source)); phases = manifest['phases']
    if len(all_bodies) != len(phases) or {i for i, _, _ in all_bodies} != set(range(len(phases))):
        raise Refusal('phase/source identity differs')
    bodies = [(i, a, b) for i, a, b in all_bodies if opening < a < b < closing]
    heads = {int(m[1]):m for m in ec.PHASE.finditer(source)}
    helpers = pp._helper_bodies(source, manifest, pairs)
    old_groups = manifest[pi.KEY]['groups']
    old_intervals = {g['interval'] for g in old_groups}
    # The old functions/calls remain byte-identical; verify their recorded ABI
    # and actual body hashes before relying on their phase exclusion.
    old_phases = set()
    for g in old_groups:
        signature = f'static __device__ __noinline__ void {g["helper"]}({PARAMETERS}) {{\n'
        if source.count(signature) != 1: raise Refusal('prior outline signature differs')
        start = source.index(signature)+len(signature); end = pairs[start-2]
        if end >= entry.start() or source[end-1] != '\n': raise Refusal('prior outline framing differs')
        if hashlib.sha256(source[start:end-1].encode()).hexdigest() != g['body_sha256']:
            raise Refusal('prior outline body identity differs')
        call = f'{g["helper"]}({ACTUALS});'
        if source[opening:closing].count(call) != 1: raise Refusal('prior outline call differs')
        old_phases.update(g['phases'])
    if {i for i, _, _ in all_bodies} - {i for i, _, _ in bodies} != old_phases:
        raise Refusal('unaccounted phase outside entry')
    scopes = [(m.start()-2, pairs[m.start()-2], m) for m in pp.OPAQUE.finditer(source)
              if source[m.start()-2:m.start()] == '{\n' and m.start()-2 in pairs
              and opening < m.start()-2 < pairs[m.start()-2] < closing]
    intervals = manifest['barrier_intervals']
    if len(scopes) not in (len(intervals), len(intervals)+1): raise Refusal('publication scope count differs')
    prior = {i for g in manifest[mixed.KEY]['groups'] for i in g['phases']}
    groups, edits, functions, refused, short, cursor = [], [], [], Counter(), Counter(), 0
    for interval, (scope_start, scope_end, opaque) in enumerate(scopes):
        members = []
        while cursor < len(bodies) and bodies[cursor][1] < scope_end:
            row = bodies[cursor]
            if not scope_start < row[1] < row[2] < scope_end: raise Refusal('phase escapes publication scope')
            members.append(row); cursor += 1
        if interval in old_intervals:
            if members: raise Refusal('prior outlined interval still contains entry phases')
            continue
        if not members: continue
        ids = [i for i, _, _ in members]
        if interval >= len(intervals) or sorted(ids) != sorted(intervals[interval]):
            raise Refusal('publication membership differs')
        try:
            if opaque[0] != 'float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n':
                raise Refusal('outer opaque ABI differs')
            scalar_names, retained = set(), set()
            for line in source[opaque.end():heads[ids[0]].start()].splitlines(keepends=True):
                if not line.strip(): continue
                scalar = pp.SCALAR.fullmatch(line)
                if scalar:
                    if pp.SCALAR_SOURCE[scalar[1]] != scalar[2] or scalar[1] in scalar_names:
                        raise Refusal('hardware-index definition differs')
                    scalar_names.add(scalar[1])
                else:
                    name, _ = pp._pointer(line)
                    if name in retained: raise Refusal('duplicate outer binding')
                    retained.add(name)
            if scalar_names != set(pp.SCALAR_SOURCE): raise Refusal('requires four exact hardware-index bindings')
            for left, right in zip(members, members[1:]):
                if source[left[2]+1:heads[right[0]].start()] != '\n__syncthreads();\n':
                    raise Refusal('interphase code differs from regular CTA barrier')
            if source[members[-1][2]+1:scope_end] != pi._publication(ids):
                raise Refusal('publication grid/tick tail differs')
        except Refusal as error:
            refused['interval: '+str(error)] += 1
            continue
        admitted, pointer_counts = [], {}
        for phase, start, end in members:
            try:
                pointer_counts[phase] = _admit(phase, source[start:end], phases[phase], prior, helpers)
                admitted.append(True)
            except Refusal as error:
                admitted.append(False); refused['phase: '+str(error)] += 1
        a = 0
        while a < len(members):
            if not admitted[a]: a += 1; continue
            b = a+1
            while b < len(members) and admitted[b]: b += 1
            if b-a < MIN_PHASES: short[b-a] += 1; a = b; continue
            selected = members[a:b]
            start, end = heads[selected[0][0]].start(), selected[-1][2]+1
            code = source[start:end]; name = f'{PREFIX}{len(groups)}'
            call = f'{name}({ACTUALS});'
            functions.append(f'static __device__ __noinline__ void {name}({PARAMETERS}) {{\n'+code+'\n}\n')
            edits.append((start, end, call, code))
            groups.append(dict(interval=interval, helper=name, phases=[i for i, _, _ in selected], phase_count=b-a,
                body_bytes=len(code.encode()), body_sha256=hashlib.sha256(code.encode()).hexdigest(),
                parameter_types=['float**', 'float*', 'unsigned', 'unsigned', 'unsigned', 'unsigned'],
                argument_names=ACTUALS.split(', '), max_phase_local_pointers=max(pointer_counts[i] for i, _, _ in selected),
                previous_phase=None if a == 0 else members[a-1][0], next_phase=None if b == len(members) else members[b][0]))
            a = b
    if cursor != len(bodies): raise Refusal('unscoped entry phase')
    pieces, inverse, cursor = [], [], 0
    for start, end, replacement, old in edits:
        if start < cursor or source[start:end] != old: raise Refusal('overlapping or changed run')
        pieces.extend((source[cursor:start], replacement)); inverse.extend((source[cursor:start], old)); cursor = end
    pieces.append(source[cursor:]); inverse.append(source[cursor:])
    if ''.join(inverse) != source: raise Refusal('run inverse failed')
    emitted = ''.join(pieces)
    emitted = emitted[:entry.start()]+''.join(functions)+emitted[entry.start():]
    result = dict(manifest)
    result[KEY] = dict(version=1, minimum_phases=MIN_PHASES, source_before=manifest['source_hash'],
        source_bytes_before=len(source.encode()), source_bytes_after=len(emitted.encode()),
        entry_body_bytes_before=entry_bytes,
        entry_body_bytes_after=entry_bytes-sum(len(old.encode())-len(new.encode()) for _, _, new, old in edits),
        outlined_runs=len(groups), outlined_intervals=len({g['interval'] for g in groups}),
        outlined_phases=sum(g['phase_count'] for g in groups), groups=groups,
        refused=dict(sorted(refused.items())), short_run_length_histogram=dict(sorted(short.items())),
        proof='complete self-contained pure runs byte-moved with exact six-value ABI; all prior helpers, caller preludes, '
              'pre/post-run CTA barriers, resources, grid/ticks/commits and prior metadata unchanged; exact inverse',
        unchanged_resource_scope='no resource/shared/private/async body moves; STEP17 guards unchanged')
    result['source_hash'] = hashlib.sha256((emitted+json.dumps(result['header_hashes'], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(emitted, result)
