"""Explicit arithmetic for long-contraction, transposed-left BF16 products.

The predicate uses only matrix layouts and dtypes. It selects the existing
mma.sync k16 body with one FP32 accumulator and one BF16 store; it is not a
new model or a default selection policy. Graph-bound readout products retain
their more specific arithmetic contract.
"""

KEY = 'matrix_accumulation_contract'
CONTRACT = 'bf16_weight_vjp_mma_k16_v1'
BODY = 'popsc_bf16'
SELECTION = 'batch1 BF16 store, transposed-left/row-major-right, K>max(M,N)'
# LATCH-MOE-EMA-FAST: the same single-chain arithmetic on the TMA warpgroup body. wgmma.m64nNk16 (shared-memory
# and register A, scale-d 0 on the first k16 step) was measured bitwise equal to the mma.sync.m16n8k16 chain
# from +0 (probes/k8_equiv_v2.cu a06: 0 of 540,672,000 and 0 of 2,162,688,000 outputs; probes/k16_ss_scale0.cu
# a16: 0 of 540,672,000), so for this contract the classes below produce identical bits. The warpgroup body is
# preferred when admitted; False restores popsc_bf16 only.
WGMMA_BODY = 'pops4b_tma'
WGMMA = True
EQUIVALENT_CLASSES = {BODY: 'bf16_mma_sync_k16', WGMMA_BODY: 'bf16_wgmma_k16'}


def bodies():
    return (WGMMA_BODY, BODY) if WGMMA else (BODY,)


def validate(value):
    if value not in (None, 'legacy', CONTRACT):
        raise ValueError(f'unknown matrix accumulation contract: {value!r}')
    return CONTRACT if value == CONTRACT else None


def applies(problem):
    """Arithmetic scope, before the existing body's layout/resource checks."""
    return (problem is not None and problem.batch == 1
            and problem.dtype == problem.out_dtype == 'bfloat16'
            and problem.out_mode == 'store'
            and problem.a.row_stride == 1 and problem.a.col_stride != 1
            and problem.b.col_stride == 1 and problem.b.row_stride != 1
            and problem.k > max(problem.m, problem.n))


def record(problem, body=BODY):
    if body not in EQUIVALENT_CLASSES:
        raise ValueError(f'matrix accumulation contract body {body!r} has no measured equivalence')
    return dict(contract=CONTRACT, body=body, shape=[problem.m, problem.n, problem.k], selection=SELECTION,
                **({'equivalence': 'wgmma k16 == mma.sync k16 chain from +0 (probes a06, a16)'} if body != BODY else {}))
