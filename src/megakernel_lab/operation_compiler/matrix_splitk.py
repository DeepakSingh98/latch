"""Opt-in narrow BF16 product arithmetic; no model or parameter-name dispatch.

This is an arithmetic contract, not a default heuristic. Each of six aligned
K intervals starts an independent FP32 mma.sync k16 accumulator. Their FP32
results are added in ascending interval order and rounded to BF16 once.
"""

KEY = 'matrix_splitk_contract'
CONTRACT = 'bf16_narrow_split6_mma_k16_v1'
BODY = 'popsc_bf16_split6'
SPLITS = 6
ALIGNMENT = 64
SELECTION = 'batch1 BF16 store, transposed-left/row-major-right, N32, K>max(M,N)'


def validate(value):
    if value not in (None, 'legacy', CONTRACT):
        raise ValueError(f'unknown matrix split-K contract: {value!r}')
    return CONTRACT if value == CONTRACT else None


def applies(problem):
    from .matrix_accumulation import applies as long_k
    return long_k(problem) and problem.n == 32


def cuts(k):
    if type(k) is not int or k <= 0 or k % ALIGNMENT:
        raise ValueError('split6 K must be a positive multiple of 64')
    chunk = ((k + SPLITS * ALIGNMENT - 1) // (SPLITS * ALIGNMENT)) * ALIGNMENT
    if (SPLITS - 1) * chunk >= k:
        raise ValueError('split6 requires six nonempty K intervals')
    return [i * chunk for i in range(SPLITS)] + [k]


def record(problem):
    boundaries = cuts(problem.k)
    return dict(contract=CONTRACT, body=BODY, shape=[problem.m, problem.n, problem.k],
                selection=SELECTION, splits=SPLITS, alignment=ALIGNMENT,
                cuts=boundaries, reduction='ascending', partial_dtype='float32',
                partial_bf16_rounding=False, final_dtype='bfloat16',
                instruction='mma.sync.m16n8k16', accumulator_initial='+0')
