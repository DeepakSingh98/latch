"""CPU-worker proofs for as_strided over a full dense storage permutation."""
import copy

import pytest
import torch

from megakernel_lab.operation_compiler.lower import LoweringError, lower


def value(shape, strides, *, alias=None, offset=0):
    return dict(shape=list(shape), strides=list(strides), dtype='bfloat16',
                alias_of=alias, storage_offset=offset, output_roles=[])


def op(name, target, args):
    return dict(name=name, target='aten.'+target, args=args, kwargs={},
                outputs=[name], inputs=[args[0]['ref']], reads=[args[0]['ref']],
                writes=[], effect='pure')


def fixture(offset=0):
    # Small analogue of a packed product -> reshape -> transpose -> chunk.
    # The transpose spans the entire source allocation exactly once.
    return dict(inputs=['source'], outputs=['chunk'], metadata={}, values={
        'source': value((6,12), (12,1)),
        'reshaped': value((2,3,2,6), (36,12,6,1), alias='source'),
        'transposed': value((2,2,3,6), (36,6,12,1), alias='reshaped'),
        'chunk': value((2,2,3,2), (36,6,12,1), alias='transposed', offset=offset)},
        operations=[
            op('reshaped','view.default',[{'ref':'source'},[2,3,2,6]]),
            op('transposed','transpose.int',[{'ref':'reshaped'},1,2]),
            op('chunk','as_strided.default',[{'ref':'transposed'},[2,2,3,2],[36,6,12,1],offset])])


@pytest.mark.parametrize('offset', [0,2,4])
def test_permuted_full_allocation_chunk_matches_pytorch_values(offset):
    graph=fixture(offset);before=copy.deepcopy(graph)
    program=lower(graph,execution='compose',vocabulary_head='off')
    layout=program.manifest['layouts']['chunk']
    assert layout['root']=='source'
    assert layout['strides']==[36,6,12,1]
    assert layout['offset']==offset
    source=torch.arange(72,dtype=torch.float32).reshape(6,12).bfloat16()
    expected=source.view(2,3,2,6).transpose(1,2).chunk(3,dim=-1)[offset//2]
    actual=torch.as_strided(source,layout['shape'],layout['strides'],layout['offset'])
    assert torch.equal(actual,expected)
    assert graph==before


def test_bounds_remain_absolute_to_root_allocation():
    graph=fixture(5)
    with pytest.raises(LoweringError,match='exceeds its source allocation'):
        lower(graph,execution='compose',vocabulary_head='off')


@pytest.mark.parametrize('strides', [(36,6,12,-1), (36,6,12,1,1)])
def test_invalid_result_strides_stay_refused(strides):
    graph=fixture();graph['operations'][-1]['args'][2]=list(strides)
    with pytest.raises(LoweringError,match='exceeds its source allocation'):
        lower(graph,execution='compose',vocabulary_head='off')


@pytest.mark.parametrize('strides', [(0,6,12,1), (40,6,12,1), (36,1,12,1)])
def test_overlapping_broadcast_or_gapped_sources_stay_refused(strides):
    graph=fixture()
    graph['inputs']=['transposed'];graph['operations']=graph['operations'][-1:]
    graph['values']={name:row for name,row in graph['values'].items() if name in ('transposed','chunk')}
    graph['values']['transposed']=value((2,2,3,6),strides)
    with pytest.raises(LoweringError,match='dense offset-0 source allocation'):
        lower(graph,execution='compose',vocabulary_head='off')


def test_partial_allocation_source_stays_refused():
    graph=fixture()
    graph['values']['partial']=value((1,2,3,6),(36,6,12,1),alias='transposed')
    graph['operations'].insert(2,op('partial','slice.Tensor',[{'ref':'transposed'},0,0,1]))
    graph['operations'][-1]['args'][0]={'ref':'partial'}
    graph['operations'][-1]['inputs']=graph['operations'][-1]['reads']=['partial']
    with pytest.raises(LoweringError,match='dense offset-0 source allocation'):
        lower(graph,execution='compose',vocabulary_head='off')


def test_rebased_input_storage_stays_refused():
    graph=fixture();graph['values']['source']['storage_offset']=8
    with pytest.raises(LoweringError,match='dense offset-0 source allocation'):
        lower(graph,execution='compose',vocabulary_head='off')


def test_existing_contiguous_as_strided_admission_keeps_layout():
    graph=fixture(2)
    graph['operations']=graph['operations'][-1:]
    graph['operations'][0]['args'][0]={'ref':'source'}
    graph['operations'][0]['inputs']=graph['operations'][0]['reads']=['source']
    graph['values']={name:row for name,row in graph['values'].items() if name in ('source','chunk')}
    graph['values']['chunk']['alias_of']='source'
    program=lower(graph,execution='compose',vocabulary_head='off')
    assert program.manifest['layouts']['chunk']['root']=='source'
    assert program.manifest['layouts']['chunk']['offset']==2
