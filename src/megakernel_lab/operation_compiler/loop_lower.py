"""Compose generic operation regions into a differentiated bounded device loop.

Every forward/VJP/optimizer body is derived by PyTorch and lowered by lower().
Only region control, loop-carried storage, cotangent accumulation and final state
commit are described here. There is no model-specific arithmetic or backward.
"""
from __future__ import annotations

import hashlib
import json
import math

from .lower import (GeneratedProgram, LoweringError, TensorLayout, _BYTES, _CTYPES,
                    _contiguous, _dtype, _index, _translation_unit)


# Scheduling/storage adaptation of P-OPS-E, pinned commit
# 578fa6ea14f12b04b85148c73cf6ef44415910e1. The donor's experimental
# switches are not the public policy: selection depends only on region semantics.
# Per-runtime counter scratch preserves independent-runtime stream safety.
LOOP_OPTIONS = {'pipeline': 'auto', 'elide_stores': True, 'inplace_commit': True,
                'phase_stages': True, 'barrier': 'counter', 'inline_stages': True,
                'partition_ctas': False, 'launch_bounds': False, 'shared_tables': False,
                'region_policy': 'auto', 'region_options': {},
                'residency': 'auto', 'persistent_counters': True, 'row_fusion': True,
                'warp_softmax': True, 'batched_reduction_loads': True,
                'defer_accumulate': True, 'joint_matrix': True, 'epilogue_fusion': True,
                'write_through': True, 'copy_bypass': False, 'defer_policy': 'earliest',
                'group_rules': False, 'member_ticks': False, 'phase_ticks': False,
                'recursive_executor': True, 'recursive_boundary_elision': False}

# P-OPS-II-D 04ca670a17bb51ca6123a5c0d9a1002a27fbc83c diagnostic settings.
# The public defaults above choose from static operation and resource evidence.
FROZEN_BASE_LOOP_OPTIONS = {
    'residency': 'two', 'persistent_counters': False, 'row_fusion': False,
    'warp_softmax': False, 'batched_reduction_loads': False, 'defer_accumulate': False,
    'joint_matrix': False, 'epilogue_fusion': False, 'write_through': False,
    'copy_bypass': False, 'group_rules': False}


def _recursive_staging_refusals(region, path, *, fp32_only=False):
    """Report known region capabilities that require the ordered composer.

    Scan bodies own internal grid barriers, and uniform conditional bodies own
    selected-path control. Neither is a flat dependency stage. Inspect nested
    children before lowering, without hiding unrelated compiler failures.
    """
    from .scan import TARGETS as scan_targets

    if hasattr(region, 'true_branch'):
        return [f'recursive staging requires flat leaves; uniform conditional region {path}']
    reasons = []
    if fp32_only:
        unsupported = sorted({value.dtype for value in getattr(region, 'values', {}).values()
                              if value.dtype not in ('float32', 'int64', 'bool')})
        if unsupported:
            reasons.append(f'recursive staging requires FP32 floating values and int64/bool state; '
                           f'region {path} has {unsupported}')
    if any(str(op.target).removeprefix('torch.ops.') in scan_targets
           for op in getattr(region, 'operations', ())):
        reasons.append(f'recursive staging cannot embed selective-scan grid barriers in region {path}')
    for name, child in getattr(region, 'regions', {}).items():
        reasons.extend(_recursive_staging_refusals(child, f'{path}/{name}', fp32_only=fp32_only))
    return reasons


def select_loop_policy(graph, *, retain_all=False, options=None):
    """Choose staged region scheduling from IR properties, never names."""
    if graph.storage_mode not in ('save_boundaries', 'recompute_boundaries', 'save_body_tape'):
        raise LoweringError(f'unknown loop storage mode {graph.storage_mode}')
    if type(graph.max_steps) is not int or not 0 <= graph.max_steps <= 1024:
        raise LoweringError('bounded loop requires static max_steps in [0,1024]')
    supplied = dict(options or {})
    unknown = set(supplied) - set(LOOP_OPTIONS)
    if unknown:
        raise LoweringError(f'unknown loop lowering options {sorted(unknown)}')
    chosen = {**LOOP_OPTIONS, **supplied}
    if type(chosen['recursive_executor']) is not bool:
        raise LoweringError('recursive_executor must be boolean')
    if type(chosen['recursive_boundary_elision']) is not bool:
        raise LoweringError('recursive_boundary_elision must be boolean')
    if chosen['pipeline'] not in ('auto', False, 'full', 'blocks'):
        raise LoweringError(f"unknown loop pipeline option {chosen['pipeline']}")
    if chosen['barrier'] not in ('cg', 'counter'):
        raise LoweringError('loop barrier must be cg or per-runtime counter')
    if chosen['residency'] not in ('auto', 'one', 'two'):
        raise LoweringError('loop residency must be auto, one or two')
    if chosen['region_policy'] not in ('auto', 'reference'):
        raise LoweringError('loop region policy must be auto or reference')
    if not isinstance(chosen['region_options'], dict):
        raise LoweringError('loop region_options must be a compiler-option mapping')
    for name in ('recursive_inline_stages', 'region_expression_inline',
                 'recursive_recompute_boundary_elision'):
        if name in chosen['region_options'] and type(chosen['region_options'][name]) is not bool:
            raise LoweringError(f'{name} must be boolean')
    forbidden = {'loop_options', 'policy', 'retain_all', 'fuse_elementwise', 'output_bindings',
                 'region_body', 'barrier_schedule', 'elide_stores', 'staged', 'partition_phases'}
    if forbidden.intersection(chosen['region_options']):
        raise LoweringError('loop region_options cannot replace region storage/control policy')
    reasons = []
    if chosen['pipeline'] is False:
        reasons.append('explicit ordered scheduling')
    if retain_all:
        reasons.append('retained diagnostic values require ordered region lifetime')
    if graph.storage_mode == 'recompute_boundaries':
        reasons.append('boundary replay uses the ordered recomputation path')
    init_saved = getattr(graph, 'init_saved', None) is not None
    regions = ('init_saved' if init_saved else 'init',
               'body_saved' if graph.storage_mode == 'save_body_tape' else 'body',
               'terminal_vjp', 'body_vjp_saved' if graph.storage_mode == 'save_body_tape' else 'body_vjp',
               'init_vjp_saved' if init_saved else 'init_vjp', 'optimizer')
    if graph.metadata.get('invariant_casts'):
        regions = ('invariant_casts', *regions)
    if any(hasattr(getattr(graph, name), 'true_branch') for name in regions):
        reasons.append('uniform conditional region requires ordered selected-path replay')
    nested = any(getattr(getattr(graph, name), 'nested_operation', False) for name in regions)
    if nested:
        reasons.append('nested loop region requires ordered recursive history lifetimes')
    if graph.max_steps == 0:
        reasons.append('zero-bound loop has no cross-iteration scheduling work')
    chosen['pipeline'] = False if reasons else ('full' if chosen['pipeline'] == 'auto' else chosen['pipeline'])
    # Keep the proven single-level pipeline unchanged. Recursive scheduling uses
    # the same staged operation ABI within each history level, with a single
    # shared per-launch counter and no reassociation of inner gradient sums.
    recursive = (chosen['recursive_executor'] and not retain_all
                 and ((nested and graph.storage_mode == 'save_body_tape')
                      or graph.storage_mode == 'recompute_boundaries')
                 and supplied.get('pipeline') is not False
                 and not any(hasattr(getattr(graph, name), 'true_branch') for name in regions)
                 and chosen['region_policy'] == 'auto')
    if recursive:
        recompute = graph.storage_mode == 'recompute_boundaries'
        refusals = [reason for name in regions
                    for reason in _recursive_staging_refusals(getattr(graph, name), name,
                                                             fp32_only=recompute)]
        if recompute:
            # New replay admission is narrower than the established saved path.
            # Capture precision='float32' produces no policy metadata. A custom
            # policy with no products, FP32 attention and exact activations is
            # also an arithmetic no-op, despite its unused BF16 compute dtype.
            metadata = graph.metadata
            declared = metadata.get('precision_policy') or {}
            plain_precision = (isinstance(declared, dict) and declared.get('products') in ([], ())
                               and declared.get('attention') == 'float32'
                               and declared.get('activation') == 'exact')
            if declared and not plain_precision:
                refusals.append('recursive staging requires an exact FP32 capture precision policy')
            if metadata.get('sequence_specs') or metadata.get('sequence_names'):
                refusals.append('recursive staging does not yet admit sequence inputs')
            if metadata.get('invariant_casts'):
                refusals.append('recursive staging does not yet admit invariant cast regions')
            optimizer_kind = metadata.get('optimizer', {}).get('kind', 'unknown')
            if optimizer_kind != 'sgd_momentum':
                refusals.append(f'recursive staging requires momentum SGD; optimizer kind {optimizer_kind}')
            unsupported = sorted({value.dtype for value in getattr(graph, 'values', {}).values()
                                  if value.dtype not in ('float32', 'int64', 'bool')})
            if unsupported:
                refusals.append(f'recursive staging requires FP32 floating values and int64/bool state; '
                                f'region loop has {unsupported}')
        reasons.extend(refusals)
        recursive = not refusals
    if recursive:
        reasons = [reason for reason in reasons if reason not in (
            'nested loop region requires ordered recursive history lifetimes',
            'boundary replay uses the ordered recomputation path')]
        if graph.storage_mode == 'recompute_boundaries':
            reasons.append('boundary replay order retained with staged direct-output regions and a shared per-launch barrier')
        else:
            reasons.append('nested history order retained with staged direct-output regions and a shared per-launch barrier')
    chosen['recursive_executor'] = recursive
    if recursive:
        # Admitted recursive regions use local stage tables and bounded
        # expression inlining. Insert defaults only after admission,
        # preserving the ordinary pipeline and explicit diagnostic overrides.
        chosen['region_options'] = {'recursive_inline_stages': True,
                                    'region_expression_inline': True,
                                    **chosen['region_options']}
        if graph.storage_mode == 'recompute_boundaries':
            chosen['region_options'].setdefault('recursive_recompute_boundary_elision', True)
        if 'recursive_boundary_elision' not in supplied:
            chosen['recursive_boundary_elision'] = True
    # The caller publishes init's carry/state together. Terminal completes all
    # export copies and final-boundary reads before any reverse replay begins,
    # so these two merges also apply to reconstructed boundaries. Explicit
    # False overrides retain each diagnostic publication path.
    chosen['recursive_boundary_elision'] = bool(recursive and chosen['recursive_boundary_elision'])
    if not recursive:
        # Executor-local controls must not reach the ordinary ordered emitter.
        # Expression inlining also applies to the existing flat staged pipeline;
        # preserve it there. Copy before filtering to retain requested diagnostics.
        chosen['region_options'] = dict(chosen['region_options'])
        chosen['region_options'].pop('recursive_inline_stages', None)
        chosen['region_options'].pop('recursive_recompute_boundary_elision', None)
        if not chosen['pipeline']:
            chosen['region_options'].pop('region_expression_inline', None)
    if not chosen['pipeline']:
        chosen['barrier'] = 'counter' if recursive else 'cg'
    return chosen, {'version': 'region-dependencies-v2', 'selected': (
                        'recursive_staged' if recursive else 'pipelined' if chosen['pipeline'] else 'ordered'),
                    'reasons': reasons or ['flat saved-boundary operation regions expose complete stage effects'],
                    'donor_commit': '578fa6ea14f12b04b85148c73cf6ef44415910e1',
                    'selection': 'static region/storage semantics; no model or workload names',
                    'requested_options': supplied}


def lower_loop(graph, *, retain_all=False, fuse_elementwise=True, options=None):
    from .lower import lower

    options, policy = select_loop_policy(graph, retain_all=retain_all, options=options)
    recursive = options['recursive_executor']
    boundary_elision = options['recursive_boundary_elision']
    if options['pipeline']:
        from .loop_pipeline import lower_pipelined
        program = lower_pipelined(graph, retain_all=retain_all,
                                  fuse_elementwise=fuse_elementwise, options=options)
        program.manifest['loop_schedule_policy'] = policy
        return program

    if graph.metadata.get('deferred_grads'):
        raise LoweringError('deferred loop weight gradients require the pipelined loop composer')
    saved_body = graph.storage_mode == 'save_body_tape'
    forward_region = 'body_saved' if saved_body else 'body'
    reverse_region = 'body_vjp_saved' if saved_body else 'body_vjp'
    init_saved = getattr(graph, 'init_saved', None) is not None
    init_name, init_vjp_name = ('init_saved', 'init_vjp_saved') if init_saved else ('init', 'init_vjp')
    region_names = (init_name, forward_region, 'terminal_vjp', reverse_region, init_vjp_name, 'optimizer')
    casts = graph.metadata.get('invariant_casts', [])
    if casts:
        region_names = ('invariant_casts', *region_names)
    modes = ('save_boundaries', 'recompute_boundaries', 'save_body_tape')
    if graph.storage_mode not in modes:
        raise LoweringError(f'unknown loop storage mode {graph.storage_mode}')
    if type(graph.max_steps) is not int or not 0 <= graph.max_steps <= 1024:
        raise LoweringError('bounded loop requires static max_steps in [0,1024]')
    metadata = graph.metadata
    parameters = tuple(metadata['parameter_names'])
    carries = tuple(metadata['carry_names'])
    states = tuple(metadata['state_names'])
    optimizer_names = tuple(metadata.get('optimizer_state_names', parameters))
    optimizer_group = 'optimizer_state' if 'optimizer_state' in graph.outputs else 'momentum'
    np, no, nc, ns = len(parameters), len(optimizer_names), len(carries), len(states)
    inputs, outputs = list(graph.inputs), list(graph.output_names())
    ni = len(inputs) - np - no - ns - 1
    values = graph.values
    for name in inputs[:np+no+ns]:
        value = values[name]
        if not TensorLayout(tuple(value.shape), tuple(value.strides), name).contiguous:
            raise LoweringError(f'loop mutable binding {name} must be contiguous before final commit')
    layouts, offsets = {}, {}
    for name in inputs + outputs:
        value = values[name]
        dtype = _dtype(value)
        if dtype not in _CTYPES:
            raise LoweringError(f'loop value {name}: unsupported {dtype}')
        layouts[name] = {'shape': list(value.shape), 'strides': list(value.strides),
                         'root': name, 'offset': 0, 'dtype': dtype}
    if layouts[inputs[-1]]['dtype'] != 'int64' or math.prod(layouts[inputs[-1]]['shape']) != 1:
        raise LoweringError('loop runtime count must be one int64 element')
    from .lower import ordered_loop_regions
    if recursive:
        from .loop_region_executor import lower_region, wrap_region
    bindings = {name: (['accumulate'] * np + ['store'] * (len(getattr(graph, name).outputs) - np)
                      if name in (reverse_region, init_vjp_name)
                      else ['store'] * len(getattr(graph, name).outputs)) for name in region_names}
    with ordered_loop_regions():
        programs = {name: lower_region(getattr(graph, name), bindings[name],
                                fuse_elementwise=fuse_elementwise, policy=options['region_policy'],
                                **options['region_options']) if recursive else
                           lower(getattr(graph, name), retain_all=retain_all,
                                fuse_elementwise=fuse_elementwise, policy=options['region_policy'],
                                **options['region_options']) for name in region_names}
    align = lambda size: (size + 63) // 64 * 64
    scratch_words = align(max(p.manifest['arena_words'] for p in programs.values()))
    top = scratch_words

    def allocate(desc, count=1):
        nonlocal top
        words = align((math.prod(desc['shape']) * _BYTES[desc['dtype']] + 3)//4)
        start = top
        top += words * count
        return start, words

    for name in outputs:
        offsets[name], _ = allocate(layouts[name])
    structured = graph.outputs
    # Use the exact stable outer ABI; all returned values have disjoint storage.
    loss_ptr = f'arena + {offsets[structured["loss"]]}LL'
    gp = [f'arena + {offsets[structured["grads"][name]]}LL' for name in parameters]
    pp = [f'arena + {offsets[structured["params"][name]]}LL' for name in parameters]
    mp = [f'arena + {offsets[structured[optimizer_group][name]]}LL' for name in optimizer_names]
    sp = [f'arena + {offsets[structured["state"][name]]}LL' for name in states]
    cp = [f'arena + {offsets[structured["carry"][name]]}LL' for name in carries]
    p = [f'args[{i}]' for i in range(np)]
    m = [f'args[{np+i}]' for i in range(no)]
    s = [f'args[{np+no+i}]' for i in range(ns)]
    data = [f'args[{np+no+ns+i}]' for i in range(ni)]
    carry_desc = [layouts[structured['carry'][name]] for name in carries]
    state_desc = [layouts[structured['state'][name]] for name in states]
    param_desc = [layouts[structured['grads'][name]] for name in parameters]
    optimizer_desc = [layouts[structured[optimizer_group][name]] for name in optimizer_names]
    # Parameter gradients accumulate across iterations in FP32. Carry (and its
    # cotangent, which PyTorch's VJP returns in the carry dtype) may be FP32 or
    # BF16; storage and copies follow the declared dtype (C-OPS2-A).
    if any(d['dtype'] != 'float32' for d in param_desc):
        raise LoweringError('loop parameter-gradient accumulation requires FP32')
    if any(d['dtype'] not in ('float32', 'bfloat16') for d in carry_desc):
        raise LoweringError('loop differentiable carry must be FP32 or BF16')
    contribution = gp if recursive else [f'arena + {allocate(d)[0]}LL' for d in param_desc]
    # P-OPS4-F: one buffer per update-invariant cast, written once per launch
    # before init and read (never written) until the last reverse/init VJP.
    cast_desc = [{'shape': list(d['shape']), 'dtype': d['dtype']} for d in casts]
    cast_buffers = [f'arena + {allocate(d)[0]}LL' for d in cast_desc]
    pw = p + cast_buffers
    cot = [[f'arena + {allocate(d)[0]}LL' for d in carry_desc] for _ in range(2)]
    keep_boundaries = graph.storage_mode != 'recompute_boundaries'
    tape_size = graph.max_steps + 1 if keep_boundaries else 3
    boundaries = [allocate(d, tape_size) for d in carry_desc + state_desc]
    body_tape = [allocate(d, max(1, graph.max_steps)) for d in metadata.get('body_tape', [])] if saved_body else []
    # C-OPS4-A sequence inputs: one full buffer per sequence (written by init),
    # one cotangent buffer per differentiable sequence (slice t written by reverse step t).
    seq_specs = metadata.get('sequence_specs', [])
    seq_desc = [{'shape': list(d['shape']), 'dtype': d['dtype']} for d in seq_specs]
    for d in seq_desc:
        if (math.prod(d['shape'][1:]) * _BYTES[d['dtype']]) % 16:
            raise LoweringError('sequence slices must be 16-byte multiples')
    seq_full = [f'arena + {allocate(d)[0]}LL' for d in seq_desc]
    init_tape = [f'arena + {allocate(d)[0]}LL' for d in metadata.get('init_tape', [])] if init_saved else []
    seq_diff = [i for i, d in enumerate(seq_specs) if d['differentiable']]
    seq_cot = [f'arena + {allocate(seq_desc[i])[0]}LL' for i in seq_diff]
    slice_words = [math.prod(d['shape'][1:]) * _BYTES[d['dtype']] // 4 for d in seq_desc]
    optimizer_data = [data[i] for i in metadata.get('optimizer_inputs', [])]

    def seq_slice(index):
        return [f'{base} + ({index}) * {words}LL' for base, words in zip(seq_full, slice_words)]

    def seq_cot_slice(index):
        return [f'{seq_cot[j]} + ({index}) * {slice_words[i]}LL' for j, i in enumerate(seq_diff)]

    def zero_unexecuted():
        lines = []
        for j, i in enumerate(seq_diff):
            words = slice_words[i]
            lines.append(f'for (long long i = (long long)count * {words}LL + (long long)blockIdx.x * blockDim.x + threadIdx.x; '
                         f'i < {graph.max_steps * words}LL; i += (long long)gridDim.x * blockDim.x) '
                         f'reinterpret_cast<unsigned int*>({seq_cot[j]})[i] = 0u;')
        return '\n'.join(lines)
    connectivity = metadata.get('gradient_connectivity')
    bool_desc = {'shape': [], 'dtype': 'bool'}
    active_params = [f'arena + {allocate(bool_desc)[0]}LL' for _ in parameters] if connectivity else []
    active_cot = [[f'arena + {allocate(bool_desc)[0]}LL' for _ in carries] for _ in range(2)] if connectivity else []

    def boundary(index):
        return [f'arena + {start}LL + ({index}) * {stride}LL' for start, stride in boundaries]

    def tape(index):
        return [f'arena + {start}LL + ({index}) * {stride}LL' for start, stride in body_tape]

    def copy_one(dest, source, desc, op='='):
        ctype = _CTYPES[desc['dtype']]
        n = math.prod(desc['shape'])
        return (f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {n}LL; '
                f'i += (long long)gridDim.x * blockDim.x) '
                f'reinterpret_cast<{ctype}*>({dest})[i] {op} reinterpret_cast<const {ctype}*>({source})[i];')

    def copy_many(dest, source, desc, op='='):
        if not len(dest) == len(source) == len(desc):
            raise LoweringError('loop storage-copy ABI mismatch')
        return '\n'.join(copy_one(d, s, v, op) for d,s,v in zip(dest, source, desc))

    helpers, functions = [], []
    for name, program in programs.items():
        mf = program.manifest
        if mf['state_writes']:
            raise LoweringError(f'loop region {name} must be pure, with no implicit commits')
        if program.helpers and program.helpers not in helpers:
            helpers.append(program.helpers)
        if recursive:
            # Init's outputs and the following explicit-state/unused-sequence
            # copies are disjoint. The caller publishes all of them together
            # before either the first body or the zero-count terminal reads.
            functions.append(wrap_region(program, f'loop_{name}', bindings[name], counter=True,
                                         final_barrier=not (boundary_elision and name == init_name)))
            continue
        copies = []
        for i, value_name in enumerate(mf['outputs']):
            desc = mf['layouts'][value_name]
            root = desc['root']
            pointer = (f'args[{mf["inputs"].index(root)}]' if root in mf['inputs']
                       else f'arena + {mf["offsets"][root]}LL')
            layout = TensorLayout(tuple(desc['shape']), tuple(desc['strides']), root, desc['offset'])
            ctype = _CTYPES[desc['dtype']]
            copies.append(
                f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel}LL; '
                f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>(out[{i}])[i] = '
                f'reinterpret_cast<const {ctype}*>({pointer})[{layout.offset}LL + {_index(layout,layout.shape)}];')
        function = (f'__device__ __noinline__ void loop_{name}(float** args, float* arena, float** out) {{\n'
                         'unsigned long long* ticks = nullptr;\n' + program.kernel_body + '\n'
                         + '\n'.join(copies) + '\ngrid.sync();\n}\n')
        functions.append(f'namespace loop_detail_{name} {{\n{program.functions}\n{function}\n}}' if program.functions else function)

    def call(name, args, outs):
        mf = programs[name].manifest
        if len(args) != len(mf['inputs']) or len(outs) != len(mf['outputs']):
            raise LoweringError(f'{name} region ABI mismatch: {len(args)}/{len(outs)} versus '
                                f'{len(mf["inputs"])}/{len(mf["outputs"])}')
        return ('{\nfloat* region_args[] = {' + ', '.join(args) + '};\n'
                + 'float* region_outputs[] = {' + ', '.join(outs) + '};\n'
                + (f'loop_detail_{name}::' if programs[name].functions and not recursive else '')
                + f'loop_{name}(region_args, arena, region_outputs'
                + (', loop_counter' if recursive else '') + ');\n}')

    def mark(i):
        return f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{i}] = stamp();'

    b0 = boundary('0')
    nested = metadata.get('nested_loop')
    inner_bad = (f' || *reinterpret_cast<const int64_t*>({data[nested["count_input"]]}) < 0'
                 f' || *reinterpret_cast<const int64_t*>({data[nested["count_input"]]}) > {nested["max_steps"]}LL'
                 if nested else '')
    body = ['cg::grid_group grid = cg::this_grid();', mark(0),
            f'const int64_t requested_count = *reinterpret_cast<const int64_t*>(args[{len(inputs)-1}]);',
            f'const int count = int(requested_count < 0 ? 0 : (requested_count > {graph.max_steps}LL ? {graph.max_steps}LL : requested_count));',
            f'if (blockIdx.x == 0 && threadIdx.x == 0) {{ '
            f'*reinterpret_cast<int64_t*>(arena + {offsets[structured["executed_count"]]}LL) = count; '
            f'*reinterpret_cast<int64_t*>(arena + {offsets[structured["status"]]}LL) = (requested_count != count{inner_bad}) ? 1 : 0; }}',
            *([call('invariant_casts', p + data, cast_buffers)] if casts else []),
            call(init_name, pw + data, b0[:nc] + seq_full + init_tape), copy_many(b0[nc:], s, state_desc), zero_unexecuted(),
            'grid.sync();', mark(1)]
    if keep_boundaries:
        before, after = boundary('iteration'), boundary('iteration+1')
    else:
        before = boundary('iteration == 0 ? 0 : (1 + ((iteration-1) % 2))')
        after = boundary('1 + (iteration % 2)')
    body += ['for (int iteration = 0; iteration < count; ++iteration) {',
             call(forward_region, pw + before[:nc] + before[nc:] + seq_slice('iteration') + data,
                  after + (tape('iteration') if saved_body else [])), '}', mark(2)]
    final_index = 'count' if keep_boundaries else 'count == 0 ? 0 : (1 + ((count-1) % 2))'
    final_boundary = boundary(final_index)
    # Recursive forward completion (or init's caller barrier when count == 0)
    # already publishes this history slice. Reading it directly lets exports
    # proceed independently of terminal arithmetic. Terminal's final barrier
    # still publishes every carry/state export before any later state commit.
    terminal_carry = final_boundary[:nc] if boundary_elision else cp
    body += [copy_many(cp+sp, final_boundary, carry_desc+state_desc),
             *([] if boundary_elision else ['grid.sync();']),
             call('terminal_vjp', pw + terminal_carry + data, [loss_ptr] + gp + cot[0]), mark(3)]
    def boolean(pointer):
        return f'(*reinterpret_cast<bool*>({pointer}))'

    if connectivity:
        body += ['if (blockIdx.x == 0 && threadIdx.x == 0) {',
                 '\n'.join(f'{boolean(ptr)} = {str(bool(value)).lower()};'
                           for ptr, value in zip(active_params + active_cot[0], connectivity['terminal'])),
                 '}', 'grid.sync();']
    body += ['for (int reverse = count - 1; reverse >= 0; --reverse) {']
    if graph.storage_mode == 'recompute_boundaries':
        rb = boundary('replay == 0 ? 0 : (1 + ((replay-1) % 2))')
        ra = boundary('1 + (replay % 2)')
        body += ['for (int replay = 0; replay < reverse; ++replay) {',
                 call('body', pw + rb[:nc] + rb[nc:] + seq_slice('replay') + data, ra), '}']
        before = boundary('reverse == 0 ? 0 : (1 + ((reverse-1) % 2))')
    else:
        before = boundary('reverse')
    # Static two buffers with a uniform runtime pointer selection avoid host state.
    current_cot = [f'((count-1-reverse) % 2 == 0 ? {a} : {b})' for a,b in zip(*cot)]
    next_cot = [f'((count-1-reverse) % 2 == 0 ? {b} : {a})' for a,b in zip(*cot)]
    body += [call(reverse_region,
                  pw + before[:nc] + before[nc:] + seq_slice('reverse') + data + (tape('reverse') if saved_body else []) + current_cot,
                  contribution + next_cot + seq_cot_slice('reverse'))]
    if not recursive:
        body += [copy_many(gp, contribution, param_desc, '+='), 'grid.sync();']
    if connectivity:
        active_current = [f'((count-1-reverse) % 2 == 0 ? {a} : {b})' for a,b in zip(*active_cot)]
        active_next = [f'((count-1-reverse) % 2 == 0 ? {b} : {a})' for a,b in zip(*active_cot)]
        def backward_activity(rows):
            def any_edge(column):
                return ' || '.join(boolean(ptr) for ptr,row in zip(active_current, rows) if row[column]) or 'false'
            return '\n'.join([f'{boolean(ptr)} |= ({any_edge(i)});' for i,ptr in enumerate(active_params)]
                             + [f'{boolean(ptr)} = ({any_edge(np+i)});' for i,ptr in enumerate(active_next)])
        body += ['if (blockIdx.x == 0 && threadIdx.x == 0) {']
        if 'conditional_body' in metadata:
            selector = before[nc + states.index(metadata['conditional_body']['predicate_state'])]
            body += [f'if ({boolean(selector)}) {{', backward_activity(connectivity['body']['true']),
                     '} else {', backward_activity(connectivity['body']['false']), '}']
        else:
            body += [backward_activity(connectivity['body'])]
        body += ['}', 'grid.sync();']
    body += ['}', mark(4)]
    initial_cot = [f'(count % 2 == 0 ? {a} : {b})' for a,b in zip(*cot)]
    body += [call(init_vjp_name, pw + data + init_tape + initial_cot + seq_cot, contribution)]
    if not recursive:
        body += [copy_many(gp, contribution, param_desc, '+='), 'grid.sync();']
    body += [mark(5)]
    if connectivity:
        active_initial = [f'(count % 2 == 0 ? {a} : {b})' for a,b in zip(*active_cot)]
        body += ['if (blockIdx.x == 0 && threadIdx.x == 0) {']
        for i, ptr in enumerate(active_params):
            edges = ' || '.join(boolean(c) for c,row in zip(active_initial, connectivity['init']) if row[i]) or 'false'
            body += [f'{boolean(ptr)} |= ({edges});']
        body += ['}', 'grid.sync();']
    body += [call('optimizer', p + m + gp + active_params + optimizer_data, pp + mp), mark(6),
             copy_many(p + m + s, pp + mp + sp, param_desc + optimizer_desc + state_desc),
             'grid.sync();', mark(7)]
    kernel_body = '\n'.join(body)
    if recursive:
        from .loop_pipeline import COUNTER_BARRIER
        counter_offset, _ = allocate({'shape': [], 'dtype': 'int64'})
        kernel_body = kernel_body.replace('grid.sync();', 'latch_loop_barrier(loop_counter);')
        kernel_body = (f'unsigned long long* loop_counter = reinterpret_cast<unsigned long long*>(arena + {counter_offset}LL);\n'
                       'latch_loop_bar_begin(loop_counter);\n' + kernel_body)
        helpers.append(COUNTER_BARRIER)
    # Helpers are whole modules protected by their own #pragma once / include guard.
    helper_source = '\n'.join(helpers)
    shared_bytes = max(program.manifest.get('shared_bytes', 17408) for program in programs.values())
    from .resources import select_program_resources
    resource = select_program_resources(shared_bytes, children=[p.manifest for p in programs.values()],
        arch=options['region_options'].get('target_arch','sm_90'))
    # Embedded staged reductions retain a finite private partial array. Their
    # standalone launch wrapper is discarded here, so carry its resource bound
    # into the actual enclosing launch as well as runtime admission metadata.
    grid_bounds = [p.manifest['max_grid_blocks'] for p in programs.values()
                   if p.manifest.get('max_grid_blocks') is not None] if recursive else []
    max_grid_blocks = min(grid_bounds) if grid_bounds else None
    source = _translation_unit(kernel_body, helpers=helper_source, functions='\n'.join(functions), shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource['resident_blocks_per_sm'],
                               max_grid_blocks=max_grid_blocks)
    header_hashes = {}
    for program in programs.values():
        for name, digest in program.manifest['header_hashes'].items():
            if name in header_hashes and header_hashes[name] != digest:
                raise LoweringError(f'loop region header identities differ: {name}')
            header_hashes[name] = digest
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    writes = dict(zip(inputs[:np+no+ns], [structured['params'][k] for k in parameters]
                      + [structured[optimizer_group][k] for k in optimizer_names]
                      + [structured['state'][k] for k in states]))
    manifest = {
        'format': 'latch-ops-cuda-loop-v1', 'source_hash': source_hash, 'header_hashes': header_hashes,
        'inputs': inputs, 'outputs': outputs, 'structured_outputs': structured,
        'input_roles': {n: {'role': values[n].role, 'parameter': values[n].parameter} for n in inputs},
        'state_writes': writes, 'layouts': layouts, 'offsets': offsets,
        'optimizer_state_schema': metadata.get('optimizer_state_schema', []),
        'arena_words': top, 'arena_bytes': 4*top, 'region_scratch_bytes': 4*scratch_words,
        'shared_bytes': shared_bytes,
        **({'max_grid_blocks': max_grid_blocks} if max_grid_blocks is not None else {}),
        'boundary_tape_bytes': sum(stride*tape_size*4 for _,stride in boundaries),
        'body_tape_bytes': sum(stride*max(1, graph.max_steps)*4 for _,stride in body_tape),
        'init_tape_bytes': sum(math.prod(d['shape']) * _BYTES[d['dtype']] for d in metadata.get('init_tape', [])) if init_saved else 0,
        'retained_all': retain_all, 'phase_count': 7,
        'loop_schedule_policy': policy, 'loop_options': options,
        'recursive_executor': recursive,
        'phases': [{'kind': n, 'names': [n]} for n in (('invariant_casts',) if casts else ()) + ('init','forward_loop','terminal_vjp','reverse_loop','init_vjp','optimizer')],
        'max_gemm_tiles': max(p.manifest['max_gemm_tiles'] for p in programs.values()),
        'regions': {n:p.manifest for n,p in programs.items()},
        'compiled_regions': list(region_names),
        'runtime_loop': {'max_steps':graph.max_steps,'count_input':inputs[-1], 'storage_mode':graph.storage_mode,
                         'count_policy':'clamp to [0,max_steps]; nonzero status exposes out-of-range request',
                         'body_backward': ('PyTorch joint VJP partition; saved forward tape, cotangent-dependent reverse operations only'
                                           if saved_body else 'PyTorch VJP; local body activations recomputed from saved/reconstructed old carry and RNG')},
        'numerical_policy': programs[forward_region].manifest['numerical_policy'],
        'execution':'one cooperative kernel; runtime forward and reverse loops; one final optimizer/RNG commit',
        'memory_plan':'disjoint stable outputs/cotangents/boundaries; shared max-sized operation-region scratch',
    }
    if casts:
        manifest.update(invariant_casts=casts, invariant_cast_bytes=sum(
            math.prod(d['shape']) * _BYTES[d['dtype']] for d in cast_desc))
    if nested:
        manifest['runtime_loop']['inner'] = {
            'max_steps': nested['max_steps'],
            'count_input': inputs[np+no+ns+nested['count_input']],
            'storage_mode': graph.storage_mode,
            'count_policy': 'clamp to [0,max_steps]; outer status reports either level clamping',
            'parameter_version': nested['parameter_version'],
        }
    if resource['has_popsb']:
        manifest['resource_policy'] = resource
    return GeneratedProgram(source, manifest, kernel_body, helper_source, '\n'.join(functions))
