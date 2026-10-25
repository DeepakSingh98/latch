"""Opt-in admission, legacy isolation, and emitted CUDA reduction contract."""
import json
import os
from pathlib import Path
import torch
from megakernel_lab.operation_compiler import capture_inference
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler import torch_row_reduction as rule


def main():
    torch.set_num_threads(1)
    base = dict(metadata={'row_reduction_contract': rule.CONTRACT}, target='aten.mean.dim',
                dtype='float32', source_dtype='float32', source_shape=(128,768),
                source_strides=(768,1), source_offset=0, dims=[1], rows=128, extent=768)
    assert rule.eligible(**base)
    assert rule.eligible(**{**base, 'target': 'aten.sum.dim_IntList'})
    rejected = [dict(metadata={}), dict(target='aten.prod.dim_int'), dict(dtype='bfloat16'),
                dict(source_dtype='bfloat16'), dict(dims=[0]), dict(rows=32), dict(extent=8192),
                dict(extent=769), dict(extent=128), dict(source_strides=(1,128)),
                dict(source_strides=(1536,1)), dict(source_offset=1)]
    for change in rejected:
        assert not rule.eligible(**{**base, **change}), change
    def fn(_parameters, x, gain):
        mean = x.square().mean(-1, keepdim=True)
        y = x*torch.rsqrt(mean+1e-6)*gain
        return dict(mean=mean, normalized=y, rounded=y.bfloat16())
    captured = capture_inference(fn, {}, (torch.zeros(128,768), torch.ones(768)))
    legacy = lower(captured.ir, target_arch='sm_90', execution='compose')
    assert rule.CONTRACT not in legacy.source
    captured.ir.metadata['row_reduction_contract'] = rule.CONTRACT
    current = lower(captured.ir, target_arch='sm_90', execution='compose')
    diagnostic = dict(metadata=captured.ir.metadata,
        operations=[row for row in captured.ir.to_dict()['operations'] if 'mean' in row['target']],
        execution_selection=current.manifest.get('execution_selection'),
        phases=[{key: p.get(key) for key in ('kind', 'names', 'targets', 'engine', 'row_arithmetic')}
                for p in current.manifest['phases']])
    print(json.dumps(diagnostic, indent=2), flush=True)
    if os.environ.get('FGM_JOB_OUT'):
        directory = Path(os.environ['FGM_JOB_OUT'])
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'norm-contract-ir.json').write_text(json.dumps(captured.ir.to_dict())+'\n')
        (directory/'norm-contract-manifest.json').write_text(json.dumps(current.manifest, default=str)+'\n')
        (directory/'norm-contract-program.cu').write_text(current.source)
    assert rule.CONTRACT in current.source
    selected = [p for p in current.manifest['phases'] if p.get('row_arithmetic') == rule.CONTRACT]
    assert len(selected) == 1, current.manifest['phases']
    assert selected[0]['engine'] == 'torch_cuda_vector4_mean'
    assert current.manifest['execution_selection']['selected'] == 'common_static'
    assert 'offset = 1; offset < 32; offset <<= 1' in current.source
    assert '__shfl_sync(0xffffffffu, second, 0)' in current.source
    assert 'latch_acc4[4]' in current.source
    del captured.ir.metadata['row_reduction_contract']
    unchanged = lower(captured.ir, target_arch='sm_90', execution='compose')
    assert unchanged.source == legacy.source
    strided = capture_inference(lambda _p, x: x.mean(-1), {}, (torch.zeros(768,128).t(),))
    strided_legacy = lower(strided.ir, target_arch='sm_90', execution='compose')
    strided.ir.metadata['row_reduction_contract'] = rule.CONTRACT
    refused = lower(strided.ir, target_arch='sm_90', execution='compose')
    assert not any(p.get('row_arithmetic') == rule.CONTRACT for p in refused.manifest['phases'])
    assert rule.CONTRACT not in refused.source
    assert refused.source == strided_legacy.source
    assert refused.manifest['execution_selection']['selected'] == strided_legacy.manifest['execution_selection']['selected']
    summed = capture_inference(lambda _p, x: x.sum(-1, keepdim=True), {}, (torch.zeros(128,768),))
    sum_legacy = lower(summed.ir, target_arch='sm_90', execution='compose')
    assert rule.CONTRACT not in sum_legacy.source
    summed.ir.metadata['row_reduction_contract'] = rule.CONTRACT
    sum_current = lower(summed.ir, target_arch='sm_90', execution='compose')
    sum_selected = [p for p in sum_current.manifest['phases'] if p.get('row_arithmetic') == rule.CONTRACT]
    assert len(sum_selected) == 1
    assert sum_selected[0]['engine'] == 'torch_cuda_vector4_sum'
    assert sum_selected[0]['arithmetic'].endswith('identity projection')
    assert rule.CONTRACT in sum_current.source and '] = second;' in sum_current.source
    assert sum_current.manifest['execution_selection']['selected'] == 'common_static'
    del summed.ir.metadata['row_reduction_contract']
    assert lower(summed.ir, target_arch='sm_90', execution='compose').source == sum_legacy.source
    sum_strided = capture_inference(lambda _p, x: x.sum(-1), {}, (torch.zeros(768,128).t(),))
    sum_strided_legacy = lower(sum_strided.ir, target_arch='sm_90', execution='compose')
    sum_strided.ir.metadata['row_reduction_contract'] = rule.CONTRACT
    sum_refused = lower(sum_strided.ir, target_arch='sm_90', execution='compose')
    assert not any(p.get('row_arithmetic') == rule.CONTRACT for p in sum_refused.manifest['phases'])
    assert sum_refused.source == sum_strided_legacy.source
    assert sum_refused.manifest['execution_selection']['selected'] == sum_strided_legacy.manifest['execution_selection']['selected']
    print(json.dumps(dict(passed=True, checks=34, contract=rule.CONTRACT,
                         default_legacy_preserved=True, positive_source_sha256=current.cache_key)))


if __name__ == '__main__':
    main()
