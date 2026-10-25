"""Step 02: vectorized elementwise phases around matrix products in flat composed programs.

Profiles of composed MLP updates (research/latch_release, gpu-diag-01) show the elementwise phases between matrix
products running as scalar grid-stride loops with 64-bit layout arithmetic on every access, several times slower
than memory bandwidth, and odd-sized tensors (odd element counts, padded matrix operands) losing every vector path.
These rules are data movement only; every element evaluates its unchanged expression:

* ``vecbf`` (P8-C ``group_rules.vector_mixed``) with ``idx32``/``idxid`` (P6-C 32-bit identity indexing, active
  only below 2**31 elements) for flat composed programs, as the composed loop planner already uses;
* a scalar tail loop after the vector loop when the element count is not a multiple of 8;
* padded matrix operands (``matrix_adapters/padding.py``) stored element by element inside the vector loop.

``ENABLED = False`` restores the previous flat composed elementwise code (ablation).
"""
from __future__ import annotations

ENABLED = True
FLAT_RULES = '+vecbf+idx32+idxid'
TAIL_GROUP = 8              # main vector loop covers a multiple of 8 elements (FP32 float4 and BF16 uint4 groups)
MIN_ELEMENTS = 4096         # the vector passes' own minimum


def flat_rules(rules):
    """Composed flat group rules with the vector/index features added."""
    if not ENABLED:
        return rules
    present = set(str(rules).split('+'))
    return rules + ''.join('+' + f for f in FLAT_RULES.strip('+').split('+') if f not in present)


def split(numel):
    """(vector elements, tail elements) for one phase."""
    if not ENABLED or numel < MIN_ELEMENTS or numel % TAIL_GROUP == 0:
        return numel, 0
    main = numel - numel % TAIL_GROUP
    return main, numel - main


def tail_loop(start, numel, body):
    return (f"for (long long i = {start}LL + (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {numel}LL; "
            f"i += (long long)gridDim.x * blockDim.x) {{\n" + "\n".join(body) + "\n}")
