"""CPU tests for the vector-resident loop schedule (lowering only; GPU numerics are in bench/check_state).

python tests/gradlev/test_vres.py  -> JSON summary; exit 1 on failure.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import torch

from megakernel_lab.operation_compiler.inference import capture_loop_inference, lower_inference
from megakernel_lab.operation_compiler.lower import LoweringError

# LATCH-V0-NEXT: the GradLev helpers are the self-contained example package (examples/gradlev/gradlev_example,
# copied from research/latch_gradlev/gl at df184a5f with provenance headers), not the donor research tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'gradlev'))


def gradlev_plan():
    from gradlev_example.latch_loop import capture
    for dims in ((768, 3072, 50257), (1024, 4096, 50257), (576, 1536, 49152)):
        cap = capture(1, 16, None, 'cpu', dims)
        prog = lower_inference(cap.ir, loop_schedule='vector_resident')
        vr = prog.manifest['vector_resident']
        assert prog.source.count('__global__') == 1
        assert vr['barriers_per_iteration'] == 5, vr['barriers_per_iteration']
        # 2 x d x m FP32 state / 132 CTAs fits next to the other shared values only for the smaller shapes
        want = [] if dims[0] * dims[1] * 8 / 132 > 150000 else ['arg7_1', 'arg8_1']
        assert sorted(vr['resident_state']) == want, (dims, vr['resident_state'])
        forms = {k: (v['form'], v['split']) for k, v in vr['products'].items()}
        assert forms == {'bmm': ('axpy', 'F'), 'bmm_1': ('axpy', 'K'), 'mm': ('dot', 'F'),
                         'mm_1': ('axpy', 'K'), 'bmm_2': ('dot', 'F')}, forms
        assert 'atomic' not in prog.source
        nores = lower_inference(cap.ir, loop_schedule='vector_resident', vres_resident_state=False)
        assert nores.manifest['vector_resident']['resident_state'] == []
    return True


def refuses_non_owner_state():
    """A body that reads the whole state matrix in a replicated reduction must be refused."""
    W = torch.randn(1, 256, 512)

    def init(p, x):
        return {'t': torch.zeros((), dtype=torch.int64), 'acc': torch.zeros(4)}

    def body(p, carry, state, x):
        v = torch.bmm(x[None, :1, :], state['W'])            # owner split F (512)
        s = state['W'].sum(dim=1)                             # reduction over the contraction dim (non-owner)
        new = state['W'] - 1e-3 * x[None, 0, :, None] * v[:, :, :]
        return {'t': carry['t'] + 1, 'acc': carry['acc'] + s[0, :4]}, {'W': new}

    def out(p, carry, x):
        return carry['acc']
    cap = capture_loop_inference(init, body, out, {'p': torch.zeros(1)}, (torch.randn(4, 256),),
                                 torch.tensor(4), max_steps=4, state={'W': W})
    try:
        lower_inference(cap.ir, loop_schedule='vector_resident')
    except LoweringError:
        return True
    raise AssertionError('non-owner state access was admitted')


def refuses_unsupported():
    def init(p, x):
        return {'h': torch.zeros(8)}

    def body(p, carry, state, x):
        return {'h': torch.cumsum(carry['h'] + x[0], 0)}, state

    def out(p, carry, x):
        return carry['h']
    cap = capture_loop_inference(init, body, out, {'p': torch.zeros(1)}, (torch.randn(4, 8),),
                                 torch.tensor(4), max_steps=4, state={})
    try:
        lower_inference(cap.ir, loop_schedule='vector_resident')
    except LoweringError:
        return True
    raise AssertionError('cumsum was admitted')


LEGACY = {'vres_head_pipeline': False, 'vres_input_prefetch': False, 'vres_rs_warp': False, 'vres_lse_registers': False,
          'vres_loop_unroll': 1, 'vres_l2_resident_bytes': 0, 'vres_align_fusion': False, 'vres_rows_pipeline': False}


def gb60_schedule():
    """GRADLEV-BLOCK-60 schedule features at three shapes and both head policies (lowering only)."""
    from gradlev_example.latch_loop import capture
    for dims in ((768, 3072, 50257), (1024, 4096, 50257), (576, 1536, 49152)):
        for head in ('mixed_gl32', 'mixed'):
            cap = capture(1, 16, None, 'cpu', dims, head=head)
            new = lower_inference(cap.ir, loop_schedule='vector_resident')
            old = lower_inference(cap.ir, loop_schedule='vector_resident', **LEGACY)
            vn, vo = new.manifest['vector_resident'], old.manifest['vector_resident']
            notes = ' '.join(vn['notes'])
            assert new.source.count('__global__') == 1 and 'atomic' not in new.source
            assert vn['barriers_per_iteration'] == vo['barriers_per_iteration'], (dims, head)
            assert 'input prefetch' in notes and 'cp_async' in new.source and 'cp_async' not in old.kernel_body
            # thread-aligned fusion and register scalars remove CTA barrier groups
            assert new.kernel_body.count('__syncthreads();') < old.kernel_body.count('__syncthreads();'), (dims, head)
            assert 'vrs_' in new.kernel_body and 'vrs_' not in old.kernel_body
            if head == 'mixed_gl32' and dims[0] % 256:
                # K = 576 is not 16-byte-chunk vectorizable per lane: two-pass head instead of a LoweringError
                assert 'head fusion skipped' in notes, notes
            elif head == 'mixed_gl32' and dims[0] // 256 * 4 > 12:
                # K = 1024: a 4-row double buffer would exceed the register budget -> historical head loop
                assert 'head pipeline off' in notes, notes
            elif head == 'mixed_gl32':
                assert 'head fusion' in notes and 'createpolicy' in new.kernel_body
            want_state = [] if dims[0] * dims[1] * 8 / 132 > 150000 else ['arg7_1', 'arg8_1']
            assert sorted(vn['resident_state']) == want_state
            if not want_state:
                # global owner state keeps the historical row emitters (no L2 hints): the loop-invariant head
                # gets the whole residency budget
                assert 'row loads of arg8_1' not in notes and 'row loads of arg7_1' not in notes, notes
                if head == 'mixed':
                    assert 'row loads of arg4_1: L2 policy hint' in notes, notes
    return True


def main():
    torch.set_num_threads(1)
    results = {}
    for fn in (gradlev_plan, refuses_non_owner_state, refuses_unsupported, gb60_schedule):
        try:
            results[fn.__name__] = {'passed': bool(fn())}
        except Exception:
            results[fn.__name__] = {'passed': False, 'traceback': traceback.format_exc()[-3000:]}
    print(json.dumps(results, indent=2))
    sys.exit(0 if all(r['passed'] for r in results.values()) else 1)


if __name__ == '__main__':
    main()
