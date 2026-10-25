"""Compose a sampled-training program (ops.sampling) into one cooperative training kernel.

Region bodies are ordinary operation graphs lowered by ``lower()``. This module adds only region calls, the
runtime-count sampling loop with ping-pong carry (and step-changed state) boundaries, the iteration-index
table, stable outputs and the final commit of parameters, optimizer state and explicit state.

Adapted from the forward-only loop composition (ops/inference_lower.py, C-OPS-I f121882295) and the bounded
training loop's final commit (ops/loop_lower.py). Row-task execution of the sampling loop is a separate
schedule (ops/sampling_rows.py) selected by graph properties.
"""
from __future__ import annotations

import hashlib
import json
import math

from .lower import (GeneratedProgram, LoweringError, TensorLayout, _BYTES, _CTYPES, _dtype, _index,
                    _translation_unit)

SAMPLED_OPTIONS = {'row_tasks': 'off', 'row_options': {}, 'phase_ticks': False, 'body_options': {},
                   'diagnostic_defines': [], 'panels': 'auto', 'panel_options': {}, 'commit': 'alias',
                   'train_options': {}, 'train_place': 'auto'}
# P8-D defaults: the panel engine is selected whenever its rule admits the step (else the recorded refusal falls
# back to the ordered schedule); committed outputs alias their bindings. panels='off', commit='copy' reproduce the
# C-OPS4-B/LATCH-OPS parent program byte for byte.
# P9-D train_place='auto' (with commit='alias'): the training region commits the aliased parameters, optimizer state
# and unchanged explicit state itself (its graph's state_writes: the scheduler's checked in-place commit, else its
# tail commit), and its other results (loss, gradients) are exposed from the region's own storage, so the
# composition copies none of them. train_place='off' reproduces the P8-D program byte for byte.


def _align(words):
    return (words + 63) // 64 * 64


def vector_copy(dest, source, numel, ctype, nbytes):
    """Grid-strided copy of a contiguous value; 16-byte vectors when the byte size allows (pointers are
    16-byte aligned: arena allocations are 256-byte aligned and the runtime checks bindings)."""
    total = numel * nbytes
    if total % 16 == 0:
        return (f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {total // 16}LL; '
                f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<uint4*>({dest})[i] = '
                f'reinterpret_cast<const uint4*>({source})[i];')
    return (f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {numel}LL; '
            f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>({dest})[i] = '
            f'reinterpret_cast<const {ctype}*>({source})[i];')


def region_function(prefix, name, program, *, ticks=False, skip=()):
    """Device function running one lowered region and copying its outputs out (grid-strided).

    With ticks=True the function takes a diagnostic timestamp pointer (CTA 0 stamps each phase boundary)."""
    manifest = program.manifest
    copies = []
    for position, value_name in enumerate(manifest['outputs']):
        if position in skip:
            continue            # P9-D: committed by the region itself or exposed from its storage
        desc = manifest['layouts'][value_name]
        root = desc['root']
        pointer = (f'args[{manifest["inputs"].index(root)}]' if root in manifest['inputs']
                   else f'arena + {manifest["offsets"][root]}LL')
        layout = TensorLayout(tuple(desc['shape']), tuple(desc['strides']), root, desc['offset'])
        ctype = _CTYPES[desc['dtype']]
        if layout.contiguous and (layout.offset * _BYTES[desc['dtype']]) % 16 == 0:
            source = f'(reinterpret_cast<const char*>({pointer}) + {layout.offset * _BYTES[desc["dtype"]]}LL)'
            copies.append(vector_copy(f'out[{position}]', source, layout.numel, ctype, _BYTES[desc['dtype']]))
            continue
        copies.append(
            f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel}LL; '
            f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>(out[{position}])[i] = '
            f'reinterpret_cast<const {ctype}*>({pointer})[{layout.offset}LL + {_index(layout, layout.shape)}];')
    function = (f'__device__ __noinline__ void {prefix}_{name}(float** args, float* arena, float** out'
                + (', unsigned long long* ticks) {\n' if ticks else ') {\nunsigned long long* ticks = nullptr;\n')
                + program.kernel_body + '\n'
                + '\n'.join(copies) + '\ngrid.sync();\n}\n')
    if program.functions:
        return f'namespace {prefix}_detail_{name} {{\n{program.functions}\n{function}\n}}'
    return function


def lower_sampled(graph, *, options=None, **region_options):
    """Ordered composition; ``options`` selects the sampling-loop schedule (see SAMPLED_OPTIONS)."""
    from .lower import lower
    chosen = {**SAMPLED_OPTIONS, **(options or {})}
    unknown = set(chosen) - set(SAMPLED_OPTIONS)
    if unknown:
        raise LoweringError(f'unknown sampled-training options {sorted(unknown)}')
    if chosen['row_tasks'] not in ('off', 'auto', 'on'):
        raise LoweringError('row_tasks must be off, auto or on')
    if chosen['panels'] not in ('off', 'auto', 'on'):
        raise LoweringError('panels must be off, auto or on')
    if chosen['commit'] not in ('copy', 'alias'):
        raise LoweringError('commit must be copy or alias')
    if type(graph.max_steps) is not int or not 0 <= graph.max_steps <= 4096:
        raise LoweringError('sampled training requires static max_steps in [0,4096]')
    m = graph.metadata
    names, opt_names, states = m['parameter_names'], m['optimizer_state_names'], m['state_names']
    carries, changes = m['carry_names'], m['body_state_changes']
    np_, no, ns, nc, ni = len(names), len(opt_names), len(states), len(carries), m['n_inputs']
    inputs, outputs = list(graph.inputs), list(graph.output_names())
    if len(inputs) != np_ + no + ns + ni + 1:
        raise LoweringError('sampled-training input ABI arity mismatch')
    values = graph.values
    layouts, offsets = {}, {}
    for name in inputs + outputs:
        value = values[name]
        dtype = _dtype(value)
        if dtype not in _CTYPES:
            raise LoweringError(f'sampled value {name}: unsupported {dtype}')
        layouts[name] = {'shape': list(value.shape), 'strides': list(value.strides), 'root': name, 'offset': 0,
                         'dtype': dtype}
    for name in inputs[:np_ + no + ns]:
        if not TensorLayout(tuple(values[name].shape), tuple(values[name].strides), name).contiguous:
            raise LoweringError(f'committed binding {name} must be contiguous')
    count = inputs[-1]
    if layouts[count]['dtype'] != 'int64' or math.prod(layouts[count]['shape']) != 1:
        raise LoweringError('sampling count must be one int64 element')
    region_names = list(graph.regions)
    if chosen['train_place'] not in ('auto', 'off'):
        raise LoweringError('train_place must be auto or off')
    s = graph.outputs
    # P8-D commit='alias': a committed output (new parameter, optimizer state or explicit state) IS its binding:
    # the train region writes it there after its last read of the old value, and no final copy pass runs.
    aliased = {}
    if chosen['commit'] == 'alias':
        committed_outputs = ([s['params'][k] for k in names] + [s['optimizer_state'][k] for k in opt_names]
                             + [s['state'][k] for k in states])
        for i, name in enumerate(committed_outputs):
            source = inputs[i]
            a, b = layouts[name], layouts[source]
            if (a['shape'], a['strides'], a['dtype']) == (b['shape'], b['strides'], b['dtype']) and name not in aliased:
                aliased[name] = i
    nw_ = len(m.get('invariant_casts', []))
    train_graph, train_writes, train_passthrough, train_elided = graph.train, {}, set(), {}
    if chosen['train_place'] == 'auto' and chosen['commit'] == 'alias' and aliased:
        import dataclasses
        tin, tout = list(graph.train.inputs), list(graph.train.outputs)
        pairs = ([(i, 1 + np_ + i, s['params'][names[i]]) for i in range(np_)]
                 + [(np_ + nw_ + i, 1 + 2 * np_ + i, s['optimizer_state'][opt_names[i]]) for i in range(no)]
                 + [(np_ + nw_ + no + nc + i, 1 + 2 * np_ + no + i, s['state'][states[i]])
                    for i in range(ns) if states[i] not in changes])
        for region_in, region_out, composed in pairs:
            if composed not in aliased:
                continue
            if tout[region_out] == tin[region_in]:
                train_passthrough.add(region_out)       # unchanged state: nothing to commit
            elif tout[region_out] not in tin and tout.count(tout[region_out]) == 1:
                train_writes[tin[region_in]] = tout[region_out]
        if train_writes and hasattr(graph.train, 'state_writes'):
            # A committed result that is only a copy (clone/detach/alias) of an otherwise unread region value is
            # replaced by that value: the region then commits the value itself (e.g. an index_copy row scatter in
            # place) instead of materializing and storing a full copy. Value-identical results.
            from .ir import Ref
            producer, readers = {}, {}
            for op in graph.train.operations:
                for o in op.outputs:
                    producer[o] = op
                for r in op.inputs:
                    readers.setdefault(r, set()).add(op.name)
            elided, new_outputs = {}, list(tout)
            for target, result in list(train_writes.items()):
                op = producer.get(result)
                if op is None or op.target not in ('aten.clone.default', 'aten.detach.default', 'aten.alias.default'):
                    continue
                inner = op.args[0].name if isinstance(op.args[0], Ref) else None
                views = ('detach', 'alias', 'view', '_unsafe_view', 'reshape', 't', 'transpose', 'permute', 'expand',
                         'slice', 'select', 'as_strided', 'unsqueeze', 'squeeze', 'clone', 'contiguous', 'lift_fresh_copy')
                inner_op = producer.get(inner)
                if (inner is None or inner_op is None or (inner_op.target.split('.') + [''])[1] in views
                        or inner in tin or inner in tout or readers.get(inner) != {op.name}
                        or readers.get(result) or graph.train.values[inner].shape != graph.train.values[result].shape
                        or graph.train.values[inner].dtype != graph.train.values[result].dtype):
                    continue
                train_writes[target] = inner
                new_outputs[tout.index(result)] = inner
                elided[result] = inner
            train_graph = dataclasses.replace(graph.train, state_writes=dict(train_writes), outputs=tuple(new_outputs))
            train_elided = elided
        else:
            train_writes = {}
    # body_options: extra lower() options for the sampling body only (e.g. the staged-region rules
    # epilogue_fusion / row_fusion / group_rules, which select the ordered schedule of _lower_serial).
    # train_options: extra lower() options for the differentiated training region only (P8-D experiments).
    programs = {name: lower(train_graph if name == 'train' else getattr(graph, name), **region_options,
                            **(chosen['body_options'] if name == 'body' else
                               chosen['train_options'] if name == 'train' else {})) for name in region_names}
    for name, program in programs.items():
        if program.manifest['state_writes'] and not (name == 'train' and program.manifest['state_writes'] == train_writes):
            raise LoweringError(f'sampled region {name} must be pure; the composition owns every commit')
    casts = m.get('invariant_casts', [])
    nw = len(casts)
    expected = {'init': (np_ + nw + ns + ni, nc), 'body': (np_ + nw + nc + ns + 1 + ni, nc + len(changes)),
                'train': (np_ + nw + no + nc + ns + ni, 1 + 2 * np_ + no + ns)}
    if casts:
        expected['invariant_casts'] = (np_ + ni, nw)
    for name, (arity, results) in expected.items():
        manifest = programs[name].manifest
        if len(manifest['inputs']) != arity or len(manifest['outputs']) != results:
            raise LoweringError(f'{name} region ABI mismatch: {len(manifest["inputs"])}/{len(manifest["outputs"])} '
                                f'versus {arity}/{results}')
    top = _align(max(program.manifest['arena_words'] for program in programs.values()))
    scratch_words = top

    def allocate(desc, copies=1):
        nonlocal top
        start = top
        top += _align((math.prod(desc['shape']) * _BYTES[desc['dtype']] + 3) // 4) * copies
        return start

    # P9-D: train results that are neither committed nor inputs are read from the train region's own storage (the
    # region runs last and its scratch is not reused afterwards), so the composition does not copy them.
    train_manifest = programs['train'].manifest
    train_skip, exposed = set(), {}
    if train_writes:
        tout = list(train_manifest['outputs'])
        committed_positions = {tout.index(v) for v in train_writes.values()} | train_passthrough
        train_skip |= committed_positions
        result_names = [s['loss']] + [s['grads'][k] for k in names]
        for position, composed in enumerate(result_names):
            desc = train_manifest['layouts'][tout[position]]
            root = desc['root']
            if composed in aliased or root not in train_manifest['offsets'] or composed in exposed:
                continue
            exposed[composed] = (int(train_manifest['offsets'][root]), desc)
            train_skip.add(position)
    out = {}
    for name in outputs:
        if name in aliased:
            layouts[name] = {**layouts[inputs[aliased[name]]], 'root': inputs[aliased[name]], 'offset': 0}
            out[name] = f'args[{aliased[name]}]'
        elif name in exposed:
            word, desc = exposed[name]
            if (desc['shape'], desc['dtype']) != (layouts[name]['shape'], layouts[name]['dtype']):
                raise LoweringError(f'exposed train result {name}: layout mismatch')
            offsets[name] = word
            layouts[name] = {'shape': list(desc['shape']), 'strides': list(desc['strides']), 'root': name,
                             'offset': int(desc['offset']), 'dtype': desc['dtype']}
            out[name] = f'arena + {word}LL'
        else:
            offsets[name] = allocate(layouts[name])
            out[name] = f'arena + {offsets[name]}LL'
    loss_ptr = out[s['loss']]
    gp = [out[s['grads'][k]] for k in names]
    pp = [out[s['params'][k]] for k in names]
    mp = [out[s['optimizer_state'][k]] for k in opt_names]
    sp = [out[s['state'][k]] for k in states]
    cp = [out[s['carry'][k]] for k in carries]
    executed, status = out[s['executed_count']], out[s['status']]
    p = [f'args[{i}]' for i in range(np_)]
    mo = [f'args[{np_ + i}]' for i in range(no)]
    st = [f'args[{np_ + no + i}]' for i in range(ns)]
    data = [f'args[{np_ + no + ns + i}]' for i in range(ni)]
    cast_desc = [{'shape': d['shape'], 'dtype': d['dtype']} for d in casts]
    cast_buffers = [f'arena + {allocate(d)}LL' for d in cast_desc]
    pw = p + cast_buffers
    init_manifest = programs['init'].manifest
    carry_desc = [init_manifest['layouts'][name] for name in init_manifest['outputs']]
    change_desc = [layouts[inputs[np_ + no + states.index(k)]] for k in changes]
    boundary = [[f'arena + {allocate(d)}LL' for d in carry_desc + change_desc] for _ in range(2)]
    ktable = allocate({'shape': [max(1, graph.max_steps)], 'dtype': 'int64'})
    row_schedule = None
    if chosen['row_tasks'] != 'off':
        from .sampling_rows import plan_row_tasks
        row_schedule = plan_row_tasks(graph, programs['body'], chosen, region_options)
    panel = None
    if chosen['panels'] != 'off' and row_schedule is None:
        # P8-D: cluster-resident row panels for the whole sampling loop (ops/sampling_panels.py).
        from .sampling_panels import PanelRefusal, plan_panels
        try:
            panel = plan_panels(graph, chosen['panel_options'])
        except PanelRefusal as exc:
            if chosen['panels'] == 'on':
                raise LoweringError(f'panel engine refused: {exc}') from exc
            chosen['panel_refusal'] = str(exc)
    if row_schedule is not None:
        rows_scratch = allocate({'shape': [row_schedule.scratch_words * row_schedule.tasks], 'dtype': 'float32'})
        rows_counters = allocate({'shape': [8 * row_schedule.tasks], 'dtype': 'int64'})
        rows_base = allocate({'shape': [row_schedule.tasks], 'dtype': 'int64'})
        carry_row_bytes = [math.prod(d['shape'][1:]) * _BYTES[d['dtype']] for d in carry_desc]

    def select(expression):
        return [f'(({expression}) % 2 == 0 ? {a} : {b})' for a, b in zip(*boundary)]

    def state_pointers(slot):
        changed = dict(zip(changes, slot[nc:]))
        return [changed.get(k, st[i]) for i, k in enumerate(states)]

    def copy(dest, source, desc):
        return vector_copy(dest, source, math.prod(desc['shape']), _CTYPES[desc['dtype']], _BYTES[desc['dtype']])

    body_ticks = 8
    train_ticks = body_ticks + max(programs['body'].manifest.get('phase_count', 1),
                                   4 * len(panel.g.products) + 2 if panel is not None else 0) + 2
    tick_words = train_ticks + programs['train'].manifest.get('phase_count', 1) + 2

    def call(name, args, results):
        prefix = f'sampled_detail_{name}::' if programs[name].functions else ''
        extra = (f', (ticks && iteration == 0) ? ticks + {body_ticks} : nullptr'
                 if chosen['phase_ticks'] and name == 'body' else
                 f', ticks ? ticks + {train_ticks} : nullptr' if chosen['phase_ticks'] and name == 'train' else '')
        return ('{\nfloat* region_args[] = {' + ', '.join(args) + '};\n'
                + 'float* region_outputs[] = {' + ', '.join(results) + '};\n'
                + f'{prefix}sampled_{name}(region_args, arena, region_outputs{extra});\n}}')

    def mark(i):
        return f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{i}] = stamp();'

    first, current, following, final = boundary[0], select('iteration'), select('iteration + 1'), select('count')
    ks = f'reinterpret_cast<float*>(reinterpret_cast<int64_t*>(arena + {ktable}LL) + iteration)'
    body = ['cg::grid_group grid = cg::this_grid();', mark(0),
            f'const int64_t requested_count = *reinterpret_cast<const int64_t*>(args[{len(inputs) - 1}]);',
            f'const int count = int(requested_count < 0 ? 0 : (requested_count > {graph.max_steps}LL ? '
            f'{graph.max_steps}LL : requested_count));',
            f'if (blockIdx.x == 0 && threadIdx.x == 0) {{ *reinterpret_cast<int64_t*>({executed}) = count; '
            f'*reinterpret_cast<int64_t*>({status}) = requested_count == count ? 0 : 1; }}',
            f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {max(1, graph.max_steps)}LL; '
            f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<int64_t*>(arena + {ktable}LL)[i] = i;']
    if casts:
        body.append(call('invariant_casts', p + data, cast_buffers))
    if row_schedule is not None:
        body.append(row_schedule.prologue(rows_counters, rows_base))
    panel_manifest = None
    pack_code = ''
    if panel is not None:
        from .sampling_panels import generate as generate_panels
        body_inputs = programs['body'].manifest['inputs']
        call_exprs = pw + boundary[0][:nc] + st + [ks] + data
        if len(body_inputs) != len(call_exprs):
            raise LoweringError('panel engine: body input arity mismatch')
        panel_function, panel_call, panel_header, panel_manifest = generate_panels(
            panel, input_exprs=dict(zip(body_inputs, call_exprs)), count_expr='count',
            first_carry=first[:nc], final_carry=final[:nc],
            ticks_expr=f'(ticks ? ticks + {body_ticks} : nullptr)')
        # Packed product operands: arena, written once per update after the invariant casts (same phase as init).
        pack_pointer = f'reinterpret_cast<panel::bf16*>(arena + {allocate({"shape": [panel_manifest["pack_elements"]], "dtype": "bfloat16"})}LL)'
        pack_code = panel_manifest['pack_code'].replace('__PANEL_PACK__', pack_pointer)
        panel_call = panel_call.replace('__PANEL_PACK__', pack_pointer)
        stage_bytes = max(16, panel_manifest['stage_bytes'])
        stage_pointer = f'reinterpret_cast<unsigned char*>(arena + {allocate({"shape": [stage_bytes // 2], "dtype": "bfloat16"})}LL)'
        panel_call = panel_call.replace('__PANEL_STAGE__', stage_pointer)
        # P9-D: per-CTA spill area of long-idle stored values (absent unless the planner selected spill).
        spill_bytes = panel_manifest.get('spill_area_bytes', 0)
        spill_pointer = (f'reinterpret_cast<unsigned char*>(arena + {allocate({"shape": [spill_bytes // 2], "dtype": "bfloat16"})}LL)'
                         if spill_bytes else 'nullptr')
        panel_call = panel_call.replace('__PANEL_SPILL__', spill_pointer)
    body += [call('init', pw + st + data, first[:nc]),
             '\n'.join(copy(d, st[states.index(k)], desc) for d, k, desc in zip(first[nc:], changes, change_desc)),
             pack_code, 'grid.sync();', mark(1)]
    if panel is not None:
        body += [panel_call, 'grid.sync();']
    elif row_schedule is None:
        body += ['for (int iteration = 0; iteration < count; ++iteration) {',
                 call('body', pw + current[:nc] + state_pointers(current) + [ks] + data, following), '}']
    else:
        body += [row_schedule.loop_code(pw=pw, first=first, boundary=boundary, state_pointers=state_pointers,
                                        data=data, ktable=ktable, scratch_offset=rows_scratch,
                                        counter_offset=rows_counters, base_offset=rows_base,
                                        carry_row_bytes=carry_row_bytes, nc=nc)]
    body += [mark(2),
             call('train', pw + mo + final[:nc] + state_pointers(final) + data, [loss_ptr] + gp + pp + mp + sp),
             mark(3),
             '\n'.join(copy(d, src, desc) for d, src, desc in zip(cp, final[:nc], carry_desc)),
             # The train region read every old binding; commit parameters, optimizer and explicit state.
             '\n'.join(copy(d, src, layouts[inputs[i]]) for i, (d, src) in enumerate(zip(p + mo + st, pp + mp + sp))
                       if d != src),
             'grid.sync();', mark(4)]
    kernel_body = '\n'.join(body)
    helpers = '\n'.join(dict.fromkeys(program.helpers for program in programs.values() if program.helpers))
    allowed_defines = {'POPSC_DIAG_SKIP_A', 'POPSC_DIAG_SKIP_B'}
    if set(chosen['diagnostic_defines']) - allowed_defines:
        raise LoweringError('unknown diagnostic define')
    if chosen['diagnostic_defines']:
        # Diagnostic builds only: numerically meaningless timing probes (see ops/kernels/popsc_mm.cuh).
        helpers = ''.join(f'#define {d}\n' for d in chosen['diagnostic_defines']) + helpers
    functions = [region_function('sampled', name, programs[name],
                                 ticks=chosen['phase_ticks'] and name in ('body', 'train'),
                                 skip=train_skip if name == 'train' else ())
                 for name in region_names if not (panel is not None and name == 'body')]
    if panel is not None:
        from .extra_ops import FIELD_HELPERS
        from .sampling_panels import HEADER as PANEL_HEADER
        # Diagnostic defines (timing probes only) precede the engine header they configure.
        helpers += '\n' + FIELD_HELPERS + '\n' + panel_manifest['defines'] + f'#include "{PANEL_HEADER}"\n'
        functions.append(panel_function)
    if row_schedule is not None:
        functions.append(row_schedule.functions)
        helpers += '\n' + row_schedule.helpers
    functions = '\n'.join(functions)
    if panel_manifest is not None and str(panel_manifest.get('products_engine', '')).startswith('wgmma'):
        # P9-D: like P-OPS5-C, generated assert() checks would serialize every wgmma (C7510); trap instead.
        from .p5c_matrix import TRAP_ASSERT
        functions = TRAP_ASSERT + functions
    shared_bytes = max([program.manifest.get('shared_bytes', 17408) for program in programs.values()]
                       + ([row_schedule.shared_bytes] if row_schedule is not None else [])
                       + ([panel.shared_bytes] if panel is not None else []))
    from .resources import select_program_resources
    resource_policy = select_program_resources(
        shared_bytes, children=[program.manifest for program in programs.values()],
        arch=region_options.get('target_arch', 'sm_90'))
    if panel is not None and resource_policy['resident_blocks_per_sm'] != 1:
        raise LoweringError('panel engine requires one CTA per SM')
    source = _translation_unit(kernel_body, helpers=helpers, functions=functions, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'],
                               cluster=panel.cluster if panel is not None else 0)
    header_hashes = {}
    for program in programs.values():
        for name, digest in program.manifest['header_hashes'].items():
            if header_hashes.setdefault(name, digest) != digest:
                raise LoweringError(f'region header identities differ: {name}')
    if panel is not None:
        header_hashes[PANEL_HEADER] = panel_header
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    writes = dict(zip(inputs[:np_ + no + ns], [s['params'][k] for k in names]
                      + [s['optimizer_state'][k] for k in opt_names] + [s['state'][k] for k in states]))
    manifest = {
        'format': 'latch-ops-cuda-sampled-v1', 'source_hash': source_hash, 'header_hashes': header_hashes,
        'inputs': inputs, 'outputs': outputs, 'structured_outputs': s,
        'input_roles': {name: {'role': values[name].role, 'parameter': values[name].parameter} for name in inputs},
        'state_writes': writes, 'layouts': layouts, 'offsets': offsets,
        'optimizer_state_schema': m.get('optimizer_state_schema', []),
        'arena_words': top, 'arena_bytes': 4 * top, 'region_scratch_bytes': 4 * scratch_words,
        'boundary_bytes': 2 * sum(math.prod(d['shape']) * _BYTES[d['dtype']] for d in carry_desc + change_desc),
        'shared_bytes': shared_bytes, 'phase_count': 4, 'tick_words': tick_words,
        'tick_layout': {'top': [0, 1, 2, 3, 4], 'body_first_iteration': body_ticks,
                        'body_phases': [{'kind': ph['kind'], 'names': ph['names'][:4], 'engine': ph.get('engine')}
                                        for ph in programs['body'].manifest.get('phases', [])],
                        'train': train_ticks,
                        'train_phases': [{'kind': ph['kind'], 'names': ph['names'][:4], 'engine': ph.get('engine')}
                                         for ph in programs['train'].manifest.get('phases', [])]},
        'target_arch': region_options.get('target_arch', 'sm_90'), 'resource_policy': resource_policy,
        'phases': [{'kind': kind, 'names': [kind]} for kind in ('init', 'sampling_loop', 'train', 'commit')],
        'max_gemm_tiles': max(program.manifest['max_gemm_tiles'] for program in programs.values()),
        'regions': {name: program.manifest for name, program in programs.items()},
        'compiled_regions': region_names, 'sampled_options': {k: v for k, v in chosen.items() if k not in ('row_refusal', 'panel_refusal')},
        'diagnostic_only': bool(chosen['diagnostic_defines']) or bool(panel_manifest and panel_manifest['diagnostic_only']),
        'row_tasks': ({k: v for k, v in row_schedule.manifest.items() if k != 'region_manifest'}
                      if row_schedule is not None else
                      {'selected': False, 'reason': chosen.get('row_refusal', 'row_tasks off')}),
        'panel_engine': (panel_manifest if panel_manifest is not None else
                         {'selected': False, 'reason': chosen.get('panel_refusal', 'panels off')}),
        'committed_aliases': sorted(aliased),
        'train_placement': {'mode': chosen['train_place'], 'region_state_writes': train_writes,
                            'in_place': programs['train'].manifest.get('in_place_commits', {}),
                            'tail_commits': programs['train'].manifest.get('copied_commits', {}),
                            'exposed_results': sorted(exposed), 'skipped_copies': sorted(train_skip),
                            'elided_result_copies': train_elided},
        'runtime_loop': {'max_steps': graph.max_steps, 'count_input': count,
                         'count_policy': 'clamp to [0,max_steps]; nonzero status exposes out-of-range request',
                         'boundaries': 'ping-pong carry and step-changed state; unchanged state read in place',
                         'iteration_index': 'int64 table [max_steps] in the arena; body reads its entry'},
        'numerical_policy': programs['train'].manifest['numerical_policy'],
        'derivatives': m['derivatives'],
        'execution': 'one cooperative kernel: casts, init, runtime-count sampling loop, train (loss, gradients, '
                     'optimizer), final commit',
        'memory_plan': 'shared max-sized region scratch; disjoint stable outputs, cast buffers and boundaries',
    }
    if casts:
        manifest['invariant_casts'] = casts
    return GeneratedProgram(source, manifest, kernel_body, helpers, functions)
