"""CPU code-generation contracts for B matrices inside the existing E scheduler.

No numerical claim follows from these checks; the GPU fixtures qualify the
emitted products and persistent-state updates separately.
"""
from __future__ import annotations

import traceback

from megakernel_lab.operation_compiler.lower import _lower_serial, _translation_unit


def _graph(m=64, n=64, k=4096):
    return {'inputs': ['a', 'b'], 'outputs': ['y'], 'values': {
        'a': {'shape': [m, k], 'dtype': 'float32'},
        'b': {'shape': [k, n], 'dtype': 'float32'},
        'y': {'shape': [m, n], 'dtype': 'float32'}},
        'operations': [{'name': 'y', 'target': 'aten.mm.default', 'args': ['a', 'b']}]}


def run():
    results = {}

    def check(name, function):
        try:
            function()
            results[name] = {'passed': True}
        except Exception:
            results[name] = {'passed': False, 'traceback': traceback.format_exc()}

    def split_scratch_protocol():
        program = _lower_serial(_graph(), barrier_schedule='dependency')
        manifest = program.manifest
        init, matrix = manifest['phases']
        assert init['kind'] == 'matrix_scratch_init' and matrix['engine'] == 'popsb_cpasync'
        assert matrix['ksplit'] > 1 and len(init['writes']) == 1
        assert init['reads'] == ['a', 'b'], 'initialization must not precede operand readiness'
        scratch = init['writes'][0]
        assert matrix['reads'] == ['a', 'b', scratch]
        assert matrix['writes'] == ['y']
        assert matrix['internal_mutations'] == [scratch]
        assert manifest['lifetimes'][scratch] == ('live', 0, 1)
        assert manifest['barrier_intervals'] == [[0], [1]]
        assert manifest['state_writes'] == {}
        assert scratch not in manifest['inputs'] + manifest['outputs']
        assert init['counter_words'] + init['partial_words'] == manifest['layouts'][scratch]['shape'][0]
        assert '/*B_VBASE*/' not in program.source
        assert manifest['resource_policy']['has_popsb']
        assert '__launch_bounds__(256, 2)' in program.source

    check('counter_initialization_dependency_and_private_scratch_lifetime', split_scratch_protocol)

    def staged_full_grid():
        program = _lower_serial(_graph(), output_bindings=['store'], region_body=True,
                                staged=True, barrier_schedule='dependency', partition_phases=True)
        manifest = program.manifest
        init_stage, matrix_stage = manifest['stages']
        assert init_stage['interval'] < matrix_stage['interval']
        assert matrix_stage['writes_outputs'] == [0]
        assert matrix_stage['gemm_tiles'] > 0 and matrix_stage['jobs'] == 0
        assert '(vrot + 0), smem)' in program.source
        assert '/*B_VBASE*/' not in program.source
        # Partition guards belong only to the inline initialization phase. The
        # matrix call uses the physical cooperative grid, with rotation only.
        stage = program.source[program.source.index('// Phase 1: y'):]
        stage = stage[:stage.index('break;')] if 'break;' in stage else stage
        assert 'pops::gemm_phase<' in stage
        assert 'const unsigned lb_' not in stage
        assert manifest['output_placement']['y']['mode'] == 'store'

    check('staged_partition_keeps_B_matrix_on_full_grid', staged_full_grid)

    def accumulation_retains_epilogue():
        program = _lower_serial(_graph(), output_bindings=['accumulate'], region_body=True,
                                staged=True, barrier_schedule='dependency', partition_phases=True)
        assert all(phase.get('engine') != 'popsb_cpasync' for phase in program.manifest['phases'])
        assert any('direct accumulated output' in decision['reason'] for decision in program.manifest['matrix_decisions'])
        assert 's101h::EpiResid' in program.source
        assert 'resource_policy' not in program.manifest

    check('direct_accumulation_keeps_existing_exact_epilogue', accumulation_retains_epilogue)

    def unsplit_and_reference():
        program = _lower_serial(_graph(), split_k=False)
        matrix, = program.manifest['phases']
        assert matrix['engine'] == 'popsb_cpasync' and matrix['ksplit'] == 1
        assert not matrix['private_scratch'] and not matrix['internal_mutations']
        reference = _lower_serial(_graph(), matrix_policy='frozen', split_k=False,
                                  exact_reductions=False, fuse_attention=False, ordered_index_gradients=False)
        assert 'pops::gemm_phase' not in reference.source and 'resource_policy' not in reference.manifest
        assert '__launch_bounds__' not in _translation_unit('// unchanged default')
        assert '__launch_bounds__(256, 2)' in _translation_unit('// B resource target', resident_blocks_per_sm=2)

    check('unsplit_needs_no_counter_and_reference_is_unchanged', unsplit_and_reference)
    return {'passed': all(result['passed'] for result in results.values()), 'tests': results,
            'scope': 'CPU structural contracts; native numerical qualification is separate'}
