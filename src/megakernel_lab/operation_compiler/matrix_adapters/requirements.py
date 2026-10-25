"""Explicit execution requirements of a matrix candidate and the program-resource check.

Every candidate states what it needs from the surrounding generated program; a scheduler admits it only
when ``check(requirements, resources)`` passes. Values are those of the pinned donor bodies (cited per field
in ``bodies.py``); they are requirements, not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Precision:
    operands: str                 # 'bfloat16' | 'float32'
    accumulate: str               # 'float32'
    result: str                   # 'bfloat16' | 'float32'
    rounding: str                 # e.g. 'one round-to-nearest BF16 rounding of the FP32 accumulator'
    order: str                    # accumulation order statement
    instruction: str              # 'wgmma' | 'mma.sync.m16n8k16' | 'fmaf' ...
    # Candidates with equal ``bitwise_class`` produce identical bits for every admitted problem (donor claims
    # that the harness re-verifies); different classes agree only within declared mixed-precision tolerance.
    bitwise_class: str
    split_k: int = 1              # >1: FP32 partial slices summed in a fixed order (reassociation)


@dataclass(frozen=True)
class Synchronization:
    threads_per_cta: int = 256                 # every body assumes the whole 256-thread CTA participates
    cta_subset: bool = True                    # may run on a CTA slice (lo, hi, den) of the grid
    requires_full_grid: bool = False           # tasks are dealt over the physical grid (no virtual subset)
    barrier_before: str = 'grid'               # operands produced by earlier phases must be visible (grid.sync)
    barrier_after: str = 'grid'                # consumers need a grid barrier (or a dependency release) after
    internal_cross_cta: str = 'none'           # 'none' | 'split-K arrival counters (last arriver reduces)'
    entry_resident: bool = False               # wgmma bodies must be inlined into the kernel entry (ptxas C7510)
    smem_mbarriers: int = 0                    # static shared mbarriers used (TMA stage ring)
    notes: str = ''


@dataclass(frozen=True)
class Requirements:
    arch: str                                  # minimum target: 'sm_90' or 'sm_90a' (wgmma/TMA)
    dynamic_shared_bytes: int                  # dynamic shared memory this body needs per CTA
    scratch_words: int = 0                     # FP32 global scratch words (split-K partials)
    counter_words: int = 0                     # zero-initialized uint32 counters restored to zero after use
    host_setup: tuple = ()                     # e.g. ('tensor_map:A', 'tensor_map:B') encoded by the Runtime
    headers: tuple = ()
    max_resident_ctas_per_sm: int = 1          # 2 only when the body was qualified at two CTAs/SM
    sync: Synchronization = field(default_factory=Synchronization)
    precision: Precision | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ProgramResources:
    """What the surrounding generated program provides (one cooperative launch)."""
    arch: str = 'sm_90a'
    threads_per_cta: int = 256
    dynamic_shared_limit: int = 227 * 1024     # per-CTA opt-in ceiling on H100 (232448 bytes)
    resident_ctas_per_sm: int = 1
    grid: int = 132
    tensor_maps: bool = True                   # Runtime encodes TMA maps for operand roots
    scratch: bool = True                       # program can allocate private global scratch
    entry_inlining: bool = True                # generator inlines phase bodies into the entry function
    allow_split_k: bool = True                 # reassociated FP32 slice sums are acceptable
    bitwise_class: str | None = None           # require this arithmetic class (None: declared tolerance)

    def to_dict(self):
        return asdict(self)


ARCH_ORDER = {'sm_90': 0, 'sm_90a': 1}
TWO_CTA_SHARED_LIMIT = 98304     # ops/resources.py whole-program two-CTA ceiling


def check(req: Requirements, res: ProgramResources):
    """Return (admitted, reasons). Reasons list every violated requirement (empty when admitted)."""
    reasons = []
    if res.arch not in ARCH_ORDER or ARCH_ORDER[res.arch] < ARCH_ORDER[req.arch]:
        reasons.append(f'needs {req.arch}; program targets {res.arch}')
    if res.threads_per_cta != req.sync.threads_per_cta:
        reasons.append(f'needs {req.sync.threads_per_cta} threads/CTA; program has {res.threads_per_cta}')
    if req.dynamic_shared_bytes > res.dynamic_shared_limit:
        reasons.append(f'needs {req.dynamic_shared_bytes} B dynamic shared; limit {res.dynamic_shared_limit}')
    if res.resident_ctas_per_sm > req.max_resident_ctas_per_sm:
        reasons.append(f'qualified at <= {req.max_resident_ctas_per_sm} CTA/SM; program runs {res.resident_ctas_per_sm}')
    if res.resident_ctas_per_sm == 2 and req.dynamic_shared_bytes > TWO_CTA_SHARED_LIMIT:
        reasons.append('two resident CTAs/SM admit at most 98304 B shared per CTA')
    if any(s.startswith('tensor_map') for s in req.host_setup) and not res.tensor_maps:
        reasons.append('needs Runtime-encoded TMA tensor maps')
    if (req.scratch_words or req.counter_words) and not res.scratch:
        reasons.append('needs private global scratch/counters')
    if req.sync.entry_resident and not res.entry_inlining:
        reasons.append('wgmma body must be inlined into the kernel entry (ptxas serializes it otherwise)')
    if req.precision is not None:
        if req.precision.split_k > 1 and not res.allow_split_k:
            reasons.append('split-K reassociation not admitted by the program precision policy')
        if res.bitwise_class is not None and req.precision.bitwise_class != res.bitwise_class:
            reasons.append(f'arithmetic class {req.precision.bitwise_class} != required {res.bitwise_class}')
    return not reasons, reasons
