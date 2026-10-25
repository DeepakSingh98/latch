from megakernel_lab.operation_compiler.exact_vocab_cache import select


def phase(width,name):
    return dict(engine='exact_vocab_stats',reduction_elements=width,shared_bytes=128,
        code=f'for(long long row=blockIdx.x;row<16384;row+=gridDim.x) {{ '
             f'latch_exact_vocab::stats_row(x,{name},{width},row,smem); }}')


def test_row_cache_never_increases_existing_launch_shared_storage():
    p=phase(50257,'stats');body=p['code']
    unchanged,selected=select([p],body,65536)
    assert unchanged==body and not selected and p['shared_bytes']==128
    changed,selected=select([p],body,122880)
    assert selected==[0] and '::stats_row_cached(' in changed
    assert p['shared_bytes']==100648<=122880
    assert p['exact_vocab_statistics_cache']['launch_shared_bytes']==122880


def test_mixed_widths_only_cache_rows_that_fit():
    phases=[phase(10001,'small'),phase(50257,'large')]
    body='\n'.join(p['code'] for p in phases)
    changed,selected=select(phases,body,32768)
    assert selected==[0]
    assert '::stats_row_cached(x,small,10001,row,smem)' in changed
    assert '::stats_row(x,large,50257,row,smem)' in changed
    assert phases[1]['shared_bytes']==128


def test_no_selection_when_rendered_call_is_absent():
    p=phase(10001,'stats')
    changed,selected=select([p],'removed',122880)
    assert changed=='removed' and not selected and p['shared_bytes']==128
