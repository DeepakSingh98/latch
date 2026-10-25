"""Explicit wrappers for completed ordinary v7 BF16 attention items.

This is not general async outlining. Only the audited item6 entry points move
behind a call; their header, named warpgroup barriers and all caller bytes remain.
"""
import hashlib
import json
import re

from . import elementwise_codegen as ec
from . import phase_pointer_codegen as pp

Refusal = pp.Refusal
KEY = 'attention_item_codegen'
HEADER = 'attention_item_wrappers.cuh'
ORIGINAL = 'attention_long_flat.cuh'
AUDITED_SHA256 = 'c9ea55fdf04212e2176e713a479247da8fc1a50e5e4504a34a87d0bf50ca21c5'
WRAPPER_SHA256 = '6235e237b0dcac34a77b4917d27b5a511dc9fcf0ca1589a345b6be5bf74c099b'
PREFIX = 'latch_attention_v7::'
MACROS = {'PFLATLONG_WGMMA': '1', 'PFLATLONG_FTZ': '1', 'PFLATLONG_DEFER_GSCALE': '1'}
FIELDS = set('q k v dout lq lk lv ld o dq dk dv lo ldq ldk ldv mask m_b m_i m_j has_mask mask_batched kn mmax mzero rm lse2 delta o32 live heads bh t scale scale_mode m_heads safe_rows'.split())
VALUE = re.compile(r'(?:pflatlong::Layout4\{[-\dL, ]+\}|(?:\(unsigned long long\*\))?v_\w+(?: \+ \d+(?:LL)?)?|nullptr|(?:A\.\w+ = )*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?:f|LL)?)\Z')


def _phase(body, metadata):
    kind = metadata['kind'].removeprefix('attention_long_')
    if kind not in ('forward', 'dq', 'kv'):
        raise Refusal('not a completed item phase')
    if (metadata.get('schedule_version') != 7 or metadata.get('recipe') != 'bf16'
            or metadata.get('pieces') != 1 or metadata.get('head_width') != 64
            or metadata.get('shared_bytes') != 103424):
        raise Refusal('only audited ordinary v7 BF16 item resources are admitted')
    bh, t = metadata.get('head_batch'), metadata.get('sequence')
    if (not isinstance(bh, int) or bh <= 0 or not isinstance(t, int) or not 256 <= t <= 4096
            or t % 128 or metadata.get('work_items') != bh*(t//64)):
        raise Refusal('item schedule dimensions differ')
    clean = re.sub(r'//[^\n]*', '', body).strip()
    tile = f'(int)(idx / {bh}LL)' if kind == 'kv' else f'{t//64-1} - (int)(idx / {bh}LL)'
    target = 'pflatlong::'+kind+'_item6'
    call = re.compile(re.escape(target)+r'<1, (true|false)>\(A, '+re.escape(f'(int)(idx % {bh}LL), {tile}, smem);'))
    found = list(call.finditer(clean))
    if len(found) != 1: raise Refusal('item target, template or actual arguments differ')
    match = found[0]
    expected = (f'for (long long r = 0;; ++r) {{\n'
                f'  const long long idx = pflatlong::snake2(r);\n'
                f'  if (idx >= {bh*(t//64)}LL) break;\n'
                f'  {match[0]}\n}}')
    if not clean.endswith(expected): raise Refusal('independent warpgroup loop differs')
    setup = clean[:-len(expected)].strip()
    if not setup.startswith('pflatlong::Args A{};'): raise Refusal('item argument object differs')
    setup = setup[len('pflatlong::Args A{};'):].strip()
    fields = []
    for line in setup.split(';'):
        if not line.strip(): continue
        m = re.fullmatch(r'\s*A\.(\w+) = (.+)', line, re.S)
        if not m or m[1] not in FIELDS or not VALUE.fullmatch(m[2]):
            raise Refusal('nonordinary argument initialization')
        fields.append(m[1])
    if len(fields) != len(set(fields)): raise Refusal('argument field reassignment')
    original = list(re.finditer(re.escape(match[0]), body))
    if len(original) != 1: raise Refusal('call is not literal source')
    return target, PREFIX+kind+'_item', original[0].start(), match[0], match[1]


def fold(program):
    from .lower import GeneratedProgram, _header_path
    source, manifest = program.source, program.manifest
    if KEY in manifest or re.search(r'\blatch_attention_v7\b', source) or HEADER in manifest.get('header_hashes', {}):
        raise Refusal('already wrapped or reserved name')
    if (manifest.get('execution_selection', {}).get('selected') != 'common_static'
            or manifest.get('barrier_schedule') != 'dependency'
            or any(manifest.get(k) for k in ('region_body', 'stages', 'dynamic_batch', 'output_placement'))):
        raise Refusal('requires ordinary standalone common Runtime')
    entries = list(ec.KERNEL.finditer(source))
    if len(entries) != 1 or not source[entries[0].end():].startswith('float** args, float* arena, unsigned long long* ticks) {\n'):
        raise Refusal('plain entry ABI differs')
    if re.search(r'\b(?:LATCH_CTRACE|POPS4B_TRACE|P9_TRACE|P9AS|latch_cargs)\b', source):
        raise Refusal('trace or constant-binding mode')
    if hashlib.sha256(_header_path(HEADER).read_bytes()).hexdigest() != WRAPPER_SHA256:
        raise Refusal('wrapper ABI/body lacks audited identity')
    headers = manifest.get('header_hashes', {})
    if (headers.get(ORIGINAL) != AUDITED_SHA256
            or hashlib.sha256(_header_path(ORIGINAL).read_bytes()).hexdigest() != AUDITED_SHA256):
        raise Refusal('original attention header lacks audited completion identity')
    wrapper_words = set(re.findall(r'\b[A-Za-z_]\w*\b', _header_path(HEADER).read_text()))
    macro_names = re.findall(r'(?m)^\s*#\s*(?:define|undef)\s+([A-Za-z_]\w*)', source)
    if wrapper_words.intersection(macro_names) or re.search(r'\bnamespace\s+pflatlong\b', source):
        raise Refusal('wrapper namespace or ABI token is redefined')
    directives = re.findall(r'(?m)^\s*#\s*(define|undef)\s+(PFLATLONG_\w+)\s*([^\n]*)', source)
    if directives != [('define', k, v) for k, v in MACROS.items()]:
        raise Refusal('attention arithmetic or schedule macro environment differs')
    include = '#include "'+ORIGINAL+'"\n'
    if source.count(include) != 1: raise Refusal('ordinary attention include differs')
    bodies = list(ec._phase_bodies(source)); phases = manifest['phases']
    if len(bodies) != len(phases) or {i for i, _, _ in bodies} != set(range(len(phases))):
        raise Refusal('phase/source identity differs')
    edits, groups = [], []
    for phase, start, end in bodies:
        if phases[phase]['kind'] not in ('attention_long_forward', 'attention_long_dq', 'attention_long_kv'): continue
        if start < entries[0].start(): raise Refusal('attention item already outside entry')
        target, replacement, offset, call, pow2 = _phase(source[start:end], phases[phase])
        edits.append((start+offset, start+offset+len(target), replacement))
        groups.append(dict(phase=phase, target=target, wrapper=replacement, pieces=1, pow2=pow2 == 'true',
                           call_sha256=hashlib.sha256(call.encode()).hexdigest()))
    direct = list(re.finditer(r'pflatlong::(?:forward|dq|kv)_item\w*\s*<', source))
    if len(direct) != len(edits): raise Refusal('unaccounted or alternate item target')
    emitted = source
    for start, end, replacement in reversed(edits): emitted = emitted[:start]+replacement+emitted[end:]
    if edits: emitted = emitted.replace(include, include+'#include "'+HEADER+'"\n', 1)
    result = dict(manifest)
    result['header_hashes'] = dict(headers)
    if edits: result['header_hashes'][HEADER] = hashlib.sha256(_header_path(HEADER).read_bytes()).hexdigest()
    result[KEY] = dict(version=1, source_before=manifest['source_hash'], outlined_items=len(groups),
        shared_wrappers=len({g['wrapper'] for g in groups}), groups=groups, original_header_sha256=AUDITED_SHA256,
        wrapper_header_sha256=result['header_hashes'].get(HEADER), source_bytes_before=len(source.encode()),
        source_bytes_after=len(emitted.encode()),
        proof='only exact completed v7 BF16 item call targets changed; original body/header, macros, Args ABI, '
              'snake2 schedule, shared partitions, named barriers, caller publication and prior metadata unchanged',
        completion='all WGMMA groups waited; cp.async drained; final named barrier IDs 1/2 with 128 participants; '
                   'no inserted CTA barrier; STEP17 guards unchanged')
    result['source_hash'] = hashlib.sha256((emitted+json.dumps(result['header_hashes'], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(emitted, result)
