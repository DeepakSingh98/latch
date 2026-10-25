"""Compose forward-only operation regions under a runtime device count.

Every region body is an ordinary operation graph lowered by ``lower()``; this
module adds only region calls, two ping-pong carry/state boundaries, stable
outputs and the final explicit-state commit. There is no reverse loop, tape,
loss or optimizer, and no model-specific arithmetic.

Adapted from C-OPS-I f121882295ecdcf094f8727e8d72f15fc51199ba. Each
region uses the integrated compiler and its whole-program resource admission.
"""
from __future__ import annotations

import hashlib
import json
import math

from .lower import (GeneratedProgram, LoweringError, TensorLayout, _BYTES, _CTYPES, _dtype, _index,
                    _translation_unit, lower)

REGIONS = ('init', 'body', 'output')


def _align(words):
    return (words + 63) // 64 * 64


def _region_function(name, program):
    """Device function running one lowered region and copying its outputs out."""
    manifest = program.manifest
    copies = []
    for position, value_name in enumerate(manifest['outputs']):
        desc = manifest['layouts'][value_name]
        root = desc['root']
        pointer = (f'args[{manifest["inputs"].index(root)}]' if root in manifest['inputs']
                   else f'arena + {manifest["offsets"][root]}LL')
        layout = TensorLayout(tuple(desc['shape']), tuple(desc['strides']), root, desc['offset'])
        ctype = _CTYPES[desc['dtype']]
        copies.append(
            f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel}LL; '
            f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>(out[{position}])[i] = '
            f'reinterpret_cast<const {ctype}*>({pointer})[{layout.offset}LL + {_index(layout, layout.shape)}];')
    function = (f'__device__ __noinline__ void inference_{name}(float** args, float* arena, float** out) {{\n'
                'unsigned long long* ticks = nullptr;\n' + program.kernel_body + '\n'
                + '\n'.join(copies) + '\ngrid.sync();\n}\n')
    if program.functions:
        return f'namespace inference_detail_{name} {{\n{program.functions}\n{function}\n}}'
    return function


def lower_inference_loop(graph, **options):
    if type(graph.max_steps) is not int or not 0 <= graph.max_steps <= 1024:
        raise LoweringError('inference loop requires static max_steps in [0,1024]')
    metadata = graph.metadata
    np_, ns = len(metadata['parameter_names']), len(metadata['state_names'])
    nc, ni, npred = len(metadata['carry_names']), metadata['n_inputs'], metadata['prediction_count']
    inputs, outputs = list(graph.inputs), list(graph.outputs)
    if len(inputs) != np_ + ns + ni + 1 or len(outputs) != npred + ns + 2:
        raise LoweringError('inference loop ABI arity mismatch')
    values = graph.values
    layouts, offsets = {}, {}
    for name in inputs + outputs:
        value = values[name]
        dtype = _dtype(value)
        if dtype not in _CTYPES:
            raise LoweringError(f'inference loop value {name}: unsupported {dtype}')
        layouts[name] = {'shape': list(value.shape), 'strides': list(value.strides),
                         'root': name, 'offset': 0, 'dtype': dtype}
    for name in inputs[np_:np_ + ns]:
        if not TensorLayout(tuple(values[name].shape), tuple(values[name].strides), name).contiguous:
            raise LoweringError(f'committed state {name} must be contiguous')
    count = inputs[-1]
    if layouts[count]['dtype'] != 'int64' or math.prod(layouts[count]['shape']) != 1:
        raise LoweringError('runtime count must be one int64 element')
    programs = {name: lower(getattr(graph, name), **options) for name in REGIONS}
    for name, program in programs.items():
        if program.manifest['state_writes']:
            raise LoweringError(f'inference region {name} must be pure; the loop owns the commit')
    expected = {'init': (np_ + ni, nc), 'body': (np_ + nc + ns + ni, nc + ns), 'output': (np_ + nc + ni, npred)}
    for name, (arity, results) in expected.items():
        manifest = programs[name].manifest
        if len(manifest['inputs']) != arity or len(manifest['outputs']) != results:
            raise LoweringError(f'{name} region ABI mismatch')
    top = _align(max(program.manifest['arena_words'] for program in programs.values()))
    scratch_words = top

    def allocate(desc):
        nonlocal top
        start = top
        top += _align((math.prod(desc['shape']) * _BYTES[desc['dtype']] + 3) // 4)
        return f'arena + {start}LL', start

    for name in outputs:
        _, offsets[name] = allocate(layouts[name])
    init_manifest, body_manifest = programs['init'].manifest, programs['body'].manifest
    carry_desc = [init_manifest['layouts'][name] for name in init_manifest['outputs']]
    state_desc = [layouts[name] for name in inputs[np_:np_ + ns]]
    for position, desc in enumerate(carry_desc):
        local = body_manifest['layouts'][body_manifest['outputs'][position]]
        if (local['shape'], local['dtype']) != (desc['shape'], desc['dtype']):
            raise LoweringError('body carry result differs from init carry')
    boundary = [[allocate(desc)[0] for desc in carry_desc + state_desc] for _ in range(2)]
    p = [f'args[{i}]' for i in range(np_)]
    s = [f'args[{np_ + i}]' for i in range(ns)]
    data = [f'args[{np_ + ns + i}]' for i in range(ni)]
    out = [f'arena + {offsets[name]}LL' for name in outputs]
    prediction_out, state_out = out[:npred], out[npred:npred + ns]
    executed, status = out[-2], out[-1]

    def select(expression):
        return [f'(({expression}) % 2 == 0 ? {a} : {b})' for a, b in zip(*boundary)]

    def copy(dest, source, desc):
        ctype = _CTYPES[desc['dtype']]
        return (f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {math.prod(desc["shape"])}LL; '
                f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>({dest})[i] = '
                f'reinterpret_cast<const {ctype}*>({source})[i];')

    def call(name, args, results):
        prefix = f'inference_detail_{name}::' if programs[name].functions else ''
        return ('{\nfloat* region_args[] = {' + ', '.join(args) + '};\n'
                + 'float* region_outputs[] = {' + ', '.join(results) + '};\n'
                + f'{prefix}inference_{name}(region_args, arena, region_outputs);\n}}')

    def mark(i):
        return f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{i}] = stamp();'

    first = [pointer for pointer in boundary[0]]
    current, following, final = select('iteration'), select('iteration + 1'), select('count')
    body = ['cg::grid_group grid = cg::this_grid();', mark(0),
            f'const int64_t requested_count = *reinterpret_cast<const int64_t*>(args[{len(inputs) - 1}]);',
            f'const int count = int(requested_count < 0 ? 0 : (requested_count > {graph.max_steps}LL ? '
            f'{graph.max_steps}LL : requested_count));',
            f'if (blockIdx.x == 0 && threadIdx.x == 0) {{ *reinterpret_cast<int64_t*>({executed}) = count; '
            f'*reinterpret_cast<int64_t*>({status}) = requested_count == count ? 0 : 1; }}',
            call('init', p + data, first[:nc]),
            '\n'.join(copy(dest, source, desc) for dest, source, desc in zip(first[nc:], s, state_desc)),
            'grid.sync();', mark(1),
            'for (int iteration = 0; iteration < count; ++iteration) {',
            call('body', p + current + data, following), '}', mark(2),
            call('output', p + final[:nc] + data, prediction_out), mark(3),
            # The output region never reads state; publish and commit the final state.
            '\n'.join(copy(dest, source, desc) for dest, source, desc in zip(state_out, final[nc:], state_desc)),
            '\n'.join(copy(dest, source, desc) for dest, source, desc in zip(s, final[nc:], state_desc)),
            'grid.sync();', mark(4)]
    kernel_body = '\n'.join(body)
    helpers = '\n'.join(dict.fromkeys(program.helpers for program in programs.values() if program.helpers))
    functions = '\n'.join(_region_function(name, programs[name]) for name in REGIONS)
    shared_bytes = max(program.manifest.get('shared_bytes', 17408) for program in programs.values())
    from .resources import select_program_resources
    resource_policy = select_program_resources(
        shared_bytes, children=[program.manifest for program in programs.values()],
        arch=options.get('target_arch', 'sm_90'),
        pops2_two_cta=any(program.manifest.get('resource_policy', {}).get('pops2_two_cta', False)
                          for program in programs.values()))
    source = _translation_unit(kernel_body, helpers=helpers, functions=functions, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    header_hashes = {}
    for program in programs.values():
        for name, digest in program.manifest['header_hashes'].items():
            if header_hashes.setdefault(name, digest) != digest:
                raise LoweringError(f'region header identities differ: {name}')
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    manifest = {
        'format': 'latch-ops-cuda-inference-loop-v1', 'source_hash': source_hash, 'header_hashes': header_hashes,
        'inputs': inputs, 'outputs': outputs, 'structured_outputs': outputs,
        'input_roles': {name: {'role': values[name].role, 'parameter': values[name].parameter} for name in inputs},
        'state_writes': dict(zip(inputs[np_:np_ + ns], outputs[npred:npred + ns])),
        'layouts': layouts, 'offsets': offsets, 'arena_words': top, 'arena_bytes': 4 * top,
        'region_scratch_bytes': 4 * scratch_words, 'shared_bytes': shared_bytes, 'phase_count': 4,
        'target_arch': options.get('target_arch', 'sm_90'),
        'resource_policy': resource_policy,
        'phases': [{'kind': kind, 'names': [kind]} for kind in ('init', 'forward_loop', 'output')],
        'max_gemm_tiles': max(program.manifest['max_gemm_tiles'] for program in programs.values()),
        'regions': {name: program.manifest for name, program in programs.items()},
        'lower_options': {key: value for key, value in options.items()},
        'runtime_loop': {'max_steps': graph.max_steps, 'count_input': count,
                         'count_policy': 'clamp to [0,max_steps]; nonzero status exposes out-of-range request',
                         'boundaries': 'two ping-pong carry/state slots; carry restarts from init each launch'},
        'numerical_policy': programs['body'].manifest['numerical_policy'],
        'execution': 'one cooperative kernel: init, runtime-count forward loop, output, final explicit-state commit',
        'memory_plan': 'shared max-sized region scratch; disjoint stable outputs and two boundary slots',
    }
    return GeneratedProgram(source, manifest, kernel_body, helpers, functions)
