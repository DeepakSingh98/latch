"""CPU-only E/Stage4 integration contracts; run on the cloud CPU worker."""
from __future__ import annotations

from dataclasses import replace
import json
import traceback

import torch

from megakernel_lab.operation_compiler.loop import BoundedLoopIR, OperationIR, capture_loop_training
from megakernel_lab.operation_compiler.loop_lower import select_loop_policy
from megakernel_lab.operation_compiler.lower import _lower_serial, lower
from megakernel_lab.operation_compiler.optim import AdamWConfig
from .test_loop import CONFIG, fixture, eager_step, _assert_outputs
from .test_nested import capture_fixture


def _capture(*, tape_policy='auto', adamw=False):
    params, inputs, state, initial, body, terminal = fixture(random=True)
    captured = capture_loop_training(initial, body, terminal, params, None, inputs,
                                    torch.tensor(3), AdamWConfig(eps=1e-4) if adamw else CONFIG,
                                    max_steps=3, state=state, tape_policy=tape_policy)
    return captured, (initial, body, terminal)


def test_partition_and_pipeline_select_without_model_names():
    optimized, functions = _capture()
    reference, _ = _capture(tape_policy='stage3')
    # The merged default extends pointwise_remat with exact cast replay. This
    # fixture is FP32, so the full output/gradient bitwise comparison below
    # still verifies equivalence to the original saved partition.
    assert optimized.ir.metadata['tape_policy'] == 'precision_remat'
    assert reference.ir.metadata['tape_policy'] == 'stage3'
    errors = []
    for count in (0,1,3):
        args = (*optimized.example_args[:-1], torch.tensor(count))
        expected = eager_step(optimized, functions, args)
        actual, baseline = optimized.graph_step(*args), reference.graph_step(*args)
        errors.append(_assert_outputs(actual, expected))
        for a,b in zip(actual,baseline):
            assert torch.equal(a.view(torch.int32),b.view(torch.int32)) if a.dtype == torch.float32 else torch.equal(a,b)
    program = lower(optimized.ir)
    assert program.manifest['loop_schedule_policy']['selected'] == 'pipelined'
    assert program.manifest['barrier_storage']['module_global_state'] is False
    assert 'latch_loop_bar_begin(loop_counter)' in program.source
    assert '__device__ unsigned long long latch_bar_counter' not in program.source
    assert program.manifest['tick_words'] > program.manifest['phase_count']
    assert len(program.manifest['profile_intervals']) == 5
    assert max(p['end_tick'] for p in program.manifest['profile_intervals']) < program.manifest['tick_words']
    # JSON key order and unrelated metadata names cannot affect generated code.
    restored = BoundedLoopIR.from_dict(json.loads(json.dumps(optimized.ir.to_dict(),sort_keys=True)))
    restored.metadata['case'] = 'not-a-recognized-model'
    assert lower(restored).source == program.source
    return {'counts':[0,1,3], 'max_error':max(errors), 'partition_bitwise':True,
            'region_stages': {name:len(value['stages']) for name,value in program.manifest['regions'].items()}}


def test_stage4_optimizer_abi_and_safe_fallbacks():
    captured,_ = _capture(adamw=True)
    program = lower(captured.ir)
    schema = captured.ir.metadata['optimizer_state_schema']
    assert len(schema) == 3*len(captured.parameter_names)
    assert program.manifest['optimizer_state_schema'] == schema
    assert 'active_p' in program.source and 'next_c' in program.source
    assert program.manifest['loop_schedule_policy']['selected'] == 'pipelined'
    assert len(program.manifest['state_writes']) == (len(captured.parameter_names)
            + len(captured.optimizer_state_names) + len(captured.state_names))
    nested,_ = capture_fixture()
    assert nested.ir.metadata['tape_policy'] == 'stage3'
    fallback = lower(nested.ir)
    assert fallback.manifest['loop_schedule_policy']['selected'] == 'ordered'
    assert 'uniform conditional' in ' '.join(fallback.manifest['loop_schedule_policy']['reasons'])
    for kwargs,ir in (({'retain_all':True},captured.ir), ({},replace(captured.ir,storage_mode='recompute_boundaries'))):
        _,policy=select_loop_policy(ir,**kwargs)
        assert policy['selected']=='ordered'
    return {'AdamW_state_count':len(schema),'nested_ordered':True,'recompute_and_diagnostics_ordered':True}


def test_zero_accumulation_keeps_rounding_and_typed_inplace():
    values = {'x': {'shape':[3],'strides':[1],'dtype':'float32'},
              'zero': {'shape':[3],'strides':[1],'dtype':'float32'}}
    graph = {'values':values,'inputs':['x'],'outputs':['zero'],'operations':[
        {'name':'zero','target':'aten.zeros_like.default','args':[{'ref':'x'}],'kwargs':{}}]}
    program = _lower_serial(graph,output_bindings=['accumulate'],region_body=True,
                            barrier_schedule='dependency',staged=True)
    assert not program.manifest['output_placement']['zero']['zero_omitted']
    assert 'v_zero[i] + r_zero' in program.kernel_body
    values = {'old':{'shape':[],'strides':[],'dtype':'int64'},'new':{'shape':[],'strides':[],'dtype':'int64'}}
    graph = {'values':values,'inputs':['old'],'outputs':['new'],'operations':[
        {'name':'new','target':'aten.add.Tensor','args':[{'ref':'old'},1],'kwargs':{}}]}
    program = _lower_serial(graph,output_bindings=['inplace:0'],region_body=True,
                            barrier_schedule='dependency',staged='phase')
    assert program.manifest['placed_outputs']==[0]
    assert program.manifest['stages'][0]['writes_outputs']==[0]
    assert program.manifest['stages'][0]['reads_inputs']==[0]
    return {'signed_zero_addition_preserved':True,'int64_inplace_counter_placed':True}


def test_identity_region_uses_copy_stage():
    # A pure region may contain only an input alias. Its caller-copy is stage 0,
    # even though there is no preceding operation interval.
    params = {'w': torch.eye(4) * .2}
    x = torch.ones(3,4) * .1
    def initial(p, x):
        return {'h': x}
    def body(p, carry, state, x):
        return {'h': carry['h'] @ p['w']}, state
    def terminal(p, carry, x):
        return carry['h'].square().mean()
    captured = capture_loop_training(initial,body,terminal,params,None,(x,),torch.tensor(2),
                                    CONFIG,max_steps=3)
    region = captured.ir.init
    identity = OperationIR({k:region.values[k] for k in region.inputs},[],region.inputs,
                           (region.inputs[-1],),dict(region.metadata))
    program = lower(replace(captured.ir,init=identity))
    assert program.manifest['regions']['init']['stages'] == []
    assert 'loop_init_s0' in program.source
    return {'zero_operation_region':True,'caller_copy_stage':0}


def check_independent_runtime_streams(program, library, example_args):
    """Cloud GPU regression: shared binary, private arenas, differing barrier counts.

    Submissions use two streams. Hardware is permitted to serialize cooperative
    grids; this proves stream-safe independent bindings, not actual GPU overlap.
    No source compilation or reference model computation occurs here.
    """
    from megakernel_lab.operation_compiler.runtime import Runtime
    if program.manifest.get('barrier_storage', {}).get('kind') != 'counter':
        return {'passed':True,'applicable':False,'reason':'program uses cooperative-group barriers'}
    originals = [arg.detach().clone() for arg in example_args]
    runtimes = [Runtime(program,[arg.clone() for arg in originals],library=library,blocks=2) for _ in range(2)]
    limit = program.manifest['runtime_loop']['max_steps']
    sequences = ((0,limit),(min(1,limit),min(2,limit)))
    def execute(runtime, counts):
        for count in counts:
            runtime.args[-1].fill_(count)
            runtime.step()
    for runtime,counts in zip(runtimes,sequences):
        execute(runtime,counts)
    torch.cuda.synchronize()
    expected = [([value.clone() for value in runtime.outputs],
                 [value.clone() for value in runtime.args]) for runtime in runtimes]
    for runtime in runtimes:
        for target,source in zip(runtime.args,originals):
            target.copy_(source)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(device=runtimes[0].device) for _ in runtimes]
    for runtime,counts,stream in zip(runtimes,sequences,streams):
        with torch.cuda.stream(stream):
            execute(runtime,counts)
    torch.cuda.synchronize()
    checks = []
    for runtime,(outputs,args) in zip(runtimes,expected):
        checks.append(all(torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),wanted.contiguous().reshape(-1).view(torch.uint8))
                          for actual,wanted in zip((*runtime.outputs,*runtime.args),(*outputs,*args))))
    return {'passed':all(checks),'applicable':True,'streams':2,'independent_arenas':True,
            'requested_blocks_per_runtime':2,'count_sequences':sequences,'bitwise_outputs_and_bindings':checks,
            'scope':'independent stream submissions; physical concurrent execution not asserted'}


def run():
    results=[]
    for test in (test_partition_and_pipeline_select_without_model_names,
                 test_stage4_optimizer_abi_and_safe_fallbacks,
                 test_zero_accumulation_keeps_rounding_and_typed_inplace,
                 test_identity_region_uses_copy_stage):
        try:
            results.append({'test':test.__name__,'passed':True,'details':test()})
        except Exception:
            results.append({'test':test.__name__,'passed':False,'error':traceback.format_exc()})
    return {'kind':'stage4-p-ops-e-integration-contract','tests':results,
            'passed':all(test['passed'] for test in results)}


if __name__=='__main__':
    result=run()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
