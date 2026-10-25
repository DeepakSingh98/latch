"""Use otherwise allocated program shared memory for exact vocabulary statistics."""
import re

HEADER='exact_vocab_cached.cuh'
CALL=re.compile(r'latch_exact_vocab::stats_row\([^;{}]+\)')


def select(phases,body,shared_bytes):
    """Replace admitted row bodies without raising the launch allocation.

    Called after every phase's storage requirement and rendered body are final.
    No extra arena, grid synchronization, or arithmetic policy is introduced.
    """
    selected=[]
    for index,phase in enumerate(phases):
        if phase.get('engine')!='exact_vocab_stats':
            continue
        width=phase.get('reduction_elements')
        if type(width) is not int or width<=9216 or width%4==0:
            continue
        # 32 reduction floats; at most three alignment-prefix BF16 elements.
        needed=128+((width+3+3)//4)*8
        if needed>shared_bytes:
            continue
        calls=CALL.findall(phase.get('code',''))
        if len(calls)!=1 or calls[0] not in body:
            continue
        old=calls[0];new=old.replace('::stats_row(', '::stats_row_cached(',1)
        body=body.replace(old,new)
        phase['code']=phase['code'].replace(old,new)
        phase['shared_bytes']=needed
        phase['exact_vocab_statistics_cache']=dict(
            kind='BF16 row in existing program shared storage',
            bytes=needed,launch_shared_bytes=shared_bytes,
            preserves='virtual FP32 alignment, all ILP1024 leaves and ordered reduction tree')
        selected.append(index)
    return body,selected
