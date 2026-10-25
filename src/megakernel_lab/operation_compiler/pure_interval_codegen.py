"""Opt-in noinline extraction of complete, independently pure intervals.

The regular publication grid barrier/tick stanza stays in the entry kernel.
Every phase-sequence byte, including local pointer derivation and CTA
barriers, moves as a whole. Caller opaque/scalar definitions remain in place. Mixed resource/Hopper intervals remain inline.
This does not enable or relax the separate STEP17 outlining machinery.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import re

from . import elementwise_codegen as ec
from . import phase_pointer_codegen as pp
from . import mixed_phase_pointer_codegen as mixed
from . import resource_phase_pointer_codegen as resource

Refusal = pp.Refusal
KEY = 'pure_interval_codegen'
PREFIX = 'latch_pi_body'
MIN_PHASES = 32


def _publication(members):
    stamps = ' '.join(f'ticks[{i+1}] = t;' for i in members)
    return ('\ngrid.sync();\nif (ticks && lsr_bx == 0 && lsr_tx == 0) '
            '{ const unsigned long long t = stamp(); ' + stamps + ' }\n')


def fold(program):
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if KEY in manifest or PREFIX in source: raise Refusal('already outlined or reserved helper name')
    if resource.KEY not in manifest or mixed.KEY not in manifest or pp.KEY in manifest:
        raise Refusal('requires preceding resource and mixed pointer passes')
    entries = list(ec.KERNEL.finditer(source))
    if (len(entries) != 1 or manifest.get('execution_selection', {}).get('selected') != 'common_static'
            or manifest.get('barrier_schedule') != 'dependency'
            or not {'lptr','lsreg'} <= set(manifest.get('group_rules',{}).get('local_pointers',()))
            or any(manifest.get(k) for k in ('region_body','stages','dynamic_batch','output_placement'))):
        raise Refusal('requires ordinary standalone static common Runtime')
    entry = entries[0]
    if not source[entry.end():].startswith('float** args, float* arena, unsigned long long* ticks) {\n'):
        raise Refusal('unrecognized plain kernel pointer ABI')
    # Trace stamps inside an interval would need caller-owned publication.
    if re.search(r'\b(?:LATCH_CTRACE|POPS4B_TRACE|latch_cargs)\b', source):
        raise Refusal('trace or constant-binding entry mode')
    definitions = re.findall(r'(?m)^[^\n]*\blatch_opq\([^;\n]*\) \{[^\n]*', source)
    if definitions != [pp.OPAQUE_HELPER] or re.search(r'(?m)^\s*#\s*define\s+latch_opq\b', source):
        raise Refusal('ordinary opaque helper differs')
    pairs = pp._braces(source)
    entry_open = source.index("{", entry.end())
    entry_close = pairs[entry_open]
    entry_bytes = len(source[entry_open+1:entry_close].encode())
    helpers = pp._helper_bodies(source, manifest, pairs)
    bodies, phases = list(ec._phase_bodies(source)), manifest['phases']
    heads = {int(m[1]):m for m in ec.PHASE.finditer(source)}
    if len(heads)!=len(bodies) or {i for i,_,_ in bodies} != set(range(len(phases))):
        raise Refusal('phase/source manifest identity differs')
    scopes = [(m.start()-2,pairs[m.start()-2],m) for m in pp.OPAQUE.finditer(source)
              if source[m.start()-2:m.start()]=='{\n' and m.start()-2 in pairs]
    intervals = manifest['barrier_intervals']
    if len(scopes) not in (len(intervals),len(intervals)+1): raise Refusal('publication scope count differs')
    prior = {i for g in manifest[mixed.KEY]['groups'] for i in g['phases']}
    groups, edits, functions, refused, cursor = [], [], [], Counter(), 0
    for interval,(opening,closing,opaque) in enumerate(scopes):
        members=[]
        while cursor<len(bodies) and bodies[cursor][1]<closing:
            row=bodies[cursor]
            if not opening<row[1]<row[2]<closing: raise Refusal('phase escapes publication scope')
            members.append(row);cursor+=1
        if not members: continue
        ids=[i for i,_,_ in members]
        if interval>=len(intervals) or sorted(ids)!=sorted(intervals[interval]):
            raise Refusal('publication membership differs')
        if len(members)<MIN_PHASES: continue
        if any(phases[i].get('kind')!='elementwise' or any(phases[i].get(k) for k in
                ('counter_effects','private_scratch','requires_full_grid','shared_bytes')) for i in ids):
            refused['non-pure interval or cross-phase resources']+=1;continue
        try:
            if not set(ids)<=prior: raise Refusal('every phase must already have proven local bindings')
            if opaque[0]!='float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n':
                raise Refusal('outer opaque base ABI differs')
            head=heads[ids[0]].start()
            scalar_names=set()
            for line in source[opaque.end():head].splitlines(keepends=True):
                if not line.strip(): continue
                match=pp.SCALAR.fullmatch(line)
                if not match or pp.SCALAR_SOURCE[match[1]]!=match[2] or match[1] in scalar_names:
                    raise Refusal('outer tensor/scalar/shared binding remains')
                scalar_names.add(match[1])
            if scalar_names != set(pp.SCALAR_SOURCE):
                raise Refusal('requires the four exact immutable hardware-index bindings')
            pos=head
            for j,(phase,start,end) in enumerate(members):
                if source[pos:heads[phase].start()] != ('' if j==0 else '\n__syncthreads();\n'):
                    raise Refusal('interphase code differs from regular CTA barrier')
                body=source[start:end]
                clean=resource._without_prior_prefix(phase,body,prior)
                prefix=body[:len(body)-len(clean)]
                declared={}
                # The existing prefix parser above proves its complete spelling;
                # here types bind validation of the literal body/helper call.
                for line in prefix.splitlines(keepends=True):
                    if pp.POINTER.fullmatch(line):
                        name,ty=pp._pointer(line);declared[name]=ty
                if not declared: raise Refusal('missing local pointer declarations')
                if 'out' in set(re.findall(r'\b[A-Za-z_]\w*\b',prefix)):
                    raise Refusal('unbound output pointer table')
                uses=pp._pure_body(phase,clean,declared,helpers)
                if not uses<=declared.keys(): raise Refusal('body references an enclosing tensor')
                if not set(re.findall(r'\blsr_(?:tx|bx|nt|nb)\b',clean))<=scalar_names:
                    raise Refusal('body references an enclosing scalar')
                pos=end+1
            if source[pos:closing]!=_publication(ids):
                raise Refusal('publication grid/tick tail differs')
            if not entry_open < opening < pos < entry_close:
                raise Refusal('interval is not inside the entry kernel')
            code=source[head:pos]
            helper=f'{PREFIX}{len(functions)}'
            signature=f'static __device__ __noinline__ void {helper}(float** args, float* arena, unsigned lsr_tx, unsigned lsr_bx, unsigned lsr_nt, unsigned lsr_nb) {{\n'
            definition=signature+code+'\n}\n'
            call=f'{helper}(args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);'
            functions.append(definition);edits.append((head,pos,call,code))
            groups.append(dict(interval=interval,helper=helper,phases=ids,phase_count=len(ids),
                body_sha256=hashlib.sha256(code.encode()).hexdigest(),body_bytes=len(code.encode()),
                parameter_types=['float**','float*','unsigned','unsigned','unsigned','unsigned'],
                argument_names=['args','arena','lsr_tx','lsr_bx','lsr_nt','lsr_nb'],
                publication_sha256=hashlib.sha256(source[pos:closing].encode()).hexdigest()))
        except Refusal as error: refused[str(error)]+=1
    if cursor!=len(bodies): raise Refusal('unscoped phase')
    pieces,inverse,cursor=[],[],0
    for start,end,replacement,old in edits:
        if start<cursor or source[start:end]!=old: raise Refusal('changed or overlapping interval')
        pieces.extend((source[cursor:start],replacement));inverse.extend((source[cursor:start],old));cursor=end
    pieces.append(source[cursor:]);inverse.append(source[cursor:])
    if ''.join(inverse)!=source: raise Refusal('interval inverse failed')
    emitted=''.join(pieces)
    # Entry offset is before every extraction; helper insertion is unambiguous.
    emitted=emitted[:entry.start()]+''.join(functions)+emitted[entry.start():]
    result=dict(manifest)
    result[KEY]=dict(version=1,minimum_phases=MIN_PHASES,source_before=manifest['source_hash'],
        source_bytes_before=len(source.encode()),source_bytes_after=len(emitted.encode()),
        outlined_intervals=len(groups),outlined_phases=sum(g['phase_count'] for g in groups),groups=groups,
        entry_body_bytes_before=entry_bytes,
        entry_body_bytes_after=entry_bytes-sum(len(old.encode())-len(new.encode()) for _,_,new,old in edits),
        refused_intervals=dict(sorted(refused.items())),
        proof='complete independently pure phase sequences byte-moved; caller opaque/scalar preludes, grid barriers/ticks/commits '
              'and all prior manifest semantics unchanged; raw pointers and exact immutable unsigned hardware indices by value; exact inverse',
        unchanged_resource_scope='no counter/private/shared/async/warpgroup body is outlined; STEP17 guards unchanged')
    result['source_hash']=hashlib.sha256((emitted+json.dumps(result['header_hashes'],sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(emitted,result)
