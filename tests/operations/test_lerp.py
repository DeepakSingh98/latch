"""Metadata admission checks; CUDA stage probe supplies numerical evidence."""
import json
from megakernel_lab.operation_compiler.lower import lower, LoweringError
from .test_lower_safety import _value, _op, _ref


def graph(weight, *, dtype='float32', target='lerp.Scalar'):
    return dict(inputs=['start','end'], outputs=['out'],
        values=dict(start=_value((2,3),dtype),end=_value((1,3),dtype),out=_value((2,3),dtype)),
        operations=[_op('out',target,[_ref('start'),_ref('end'),weight])])


def run():
    checks=[]
    for schedule,options in [('auto',{}),('compose',{'execution':'compose'}),
                             ('reference',{'policy':'reference'})]:
        for weight in (0.,.05,.95,-.05,-.95,1.,1.5,.499999999,-.499999999):
            program=lower(graph(weight),**options)
            assert 'fmaf(' in program.source
            # Values just inside the Python-double threshold round to +/-0.5 in
            # CUDA opmath; they must take the large-weight branch.
            if weight in (.499999999,-.499999999):
                assert 'fmaf(-(' in program.source
            assert program.manifest['layouts']['out']['shape']==[2,3]
            checks.append(dict(schedule=schedule,weight=weight,passed=True))
        for bad in (graph(float('inf')),graph(.05,dtype='bfloat16'),
                    graph(.05,target='lerp.Tensor'),graph(True),graph(1e99)):
            try:
                lower(bad,**options)
            except LoweringError:
                checks.append(dict(schedule=schedule,refusal=True,passed=True))
            else:
                raise AssertionError('unadmitted lerp signature lowered')
    return dict(schema='latch-lerp-metadata-v1',checks=checks,passed=True)


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
