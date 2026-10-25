"""Padded storage for BF16 matrix operands whose leading dimension breaks the 16-byte stride contract.

Every fast matrix body (TMA/cp.async wgmma, cp.async mma.sync) stages 16-byte operand chunks, so a BF16 operand
whose rows start at addresses that are not 16-byte aligned (a contiguous last dimension that is not a multiple of
8 elements) is refused and falls back to the scalar 16x16 SIMT tile. The logical tensor is unchanged; only the
distance between its rows in the arena is.

``plan`` selects values that the compiler itself stores, whose every reader is a matrix product (directly or
through transpose views), and gives each a row pitch rounded up to 8 elements. The producing elementwise phase
writes element ``i`` of the logical row-major tensor at ``(i / d) * pitch + i % d``; matrix bodies read the
logical extents through the padded strides, so padding elements are never read (TMA dimensions and cp.async
source sizes cover only logical elements) and never written. Arithmetic and logical shapes are unchanged.

Selection reads operation properties only (operation kinds, dtypes, shapes, reader sets); never names a model.
"""
from __future__ import annotations

import math

PITCH_ELEMENTS = 8          # 16 bytes of BF16: rows not starting on 16 bytes are refused by every fast body
PITCH_ALIGN = 64            # row pitch rounded to 64 elements = 128-byte rows (one TMA swizzle span; measured explore-04)
PAD_ALIGNED_ROWS = False    # also pad rows that are 16-byte but not PITCH_ALIGN aligned (diagnostic)
VIEW_BASES = ('t', 'transpose', 'permute', 'detach', 'alias')
MATRIX_BASES = ('mm', 'bmm')
PRODUCER_BASES = ('_to_copy',)
PRODUCT_BASES = ('mm',)
# Readers of a padded product result: generic elementwise expressions (layout-indexed reads), matrix products and
# transpose views. Everything else keeps the contiguous result.
POINTWISE_READERS = ('_to_copy', 'add', 'sub', 'mul', 'div', 'neg', 'pow', 'silu', 'silu_backward', 'sigmoid',
                     'sigmoid_backward', 'tanh', 'tanh_backward', 'relu', 'threshold_backward')
PAD_PRODUCT_OUTPUTS = True  # BF16 results of products the masked TMA body takes (16-byte stores into the padding)
MIN_ROW = 16                # rows narrower than one MMA k16 step keep their layout
ENABLED = True              # process-local ablation switch (the frozen/unpadded control build)


def _base(op, field):
    return str(field(op, 'target', field(op, 'op'))).removeprefix('torch.ops.').removeprefix('aten.').split('.')[0]


def pitch(last):
    return -(-last // PITCH_ALIGN) * PITCH_ALIGN


def strides(shape, row_pitch):
    """Row-major strides of ``shape`` whose innermost rows are ``row_pitch`` elements apart."""
    result, stride = [1], row_pitch
    for extent in reversed(shape[:-1]):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def storage_elements(shape, layout_strides):
    """Elements spanned by a positive-stride layout (largest addressed element + 1)."""
    if not shape or 0 in shape:
        return math.prod(shape)
    return 1 + sum((d - 1) * s for d, s in zip(shape, layout_strides))


def store_index(shape, row_pitch, div, mod, index='i'):
    """Storage index of logical flat element ``index`` (no ';' so store-splitting passes keep the line intact)."""
    last = shape[-1]
    return f'{div(index, last)} * {row_pitch}LL + {mod(index, last)}'


def plan(operations, values, names, *, ref, field, output_name, flatten_refs, dtype_of, protected, absorbed=(),
         excluded=(), slice_views=()):
    """Return ``{value: {'pitch': p, 'shape': [...], 'readers': [...]}}`` for padded BF16 operand values.

    ``protected`` values (outputs, state, caller placements) keep their layouts; products in ``absorbed`` belong to
    attention/loop planners that require their own source layouts, so their operands are not padded.
    """
    if not ENABLED:
        return {}
    producers = {output_name(op): op for op in operations}
    slice_views = frozenset(slice_views)
    readers = {}
    for op in operations:
        for name in dict.fromkeys(flatten_refs((list(field(op, 'args', [])), dict(field(op, 'kwargs', {}))), names)):
            readers.setdefault(name, []).append(op)

    def layout_view(op, out):
        base = _base(op, field)
        if base in VIEW_BASES:
            return True
        # Explicit chunked matrix contracts may retain their source's padded
        # pitch through unit-step slices. The lowerer's view interpreter
        # computes offsets from the actual physical layout. This is opt-in;
        # ordinary graph selection does not acquire new padding candidates.
        if base != 'slice' or out not in slice_views:
            return False
        args = list(field(op, 'args', []))
        kwargs = dict(field(op, 'kwargs', {}))
        source = ref(args[0], names) if args else None
        step = args[4] if len(args) > 4 else kwargs.get('step', 1)
        return (type(step) is int and step == 1 and source in values and out in values
                and len(field(values[source], 'shape')) == len(field(values[out], 'shape')))

    def matrix_only(name, depth=0):
        """Every use of ``name`` is a matrix operand, directly or through rank-preserving transpose views."""
        uses = readers.get(name, [])
        if not uses or depth > 4:
            return False, []
        found = []
        for op in uses:
            base = _base(op, field)
            out = output_name(op)
            if out in absorbed:
                return False, []        # the product belongs to another planner (attention/loop groups)
            if base in MATRIX_BASES:
                args = list(field(op, 'args', []))
                if len(args) < 2 or not any(ref(a, names) == name for a in args[:2]):
                    return False, []
                found.append(out)
            elif layout_view(op, out) and out not in protected:
                ok, sub = matrix_only(out, depth + 1)
                if not ok:
                    return False, []
                found.extend(sub)
            else:
                return False, []
        return True, found

    def readers_ok(name, depth=0):
        uses = readers.get(name, [])
        if not uses or depth > 4:
            return False
        for op in uses:
            base, out = _base(op, field), output_name(op)
            if out in absorbed:
                return False
            if layout_view(op, out) and out not in protected:
                if not readers_ok(out, depth + 1):
                    return False
            elif base not in MATRIX_BASES + POINTWISE_READERS:
                return False
        return True

    chosen = {}
    for name, op in producers.items():
        base = _base(op, field)
        if (PAD_PRODUCT_OUTPUTS and base in PRODUCT_BASES and name not in protected and name not in excluded
                and name not in absorbed and name in values and dtype_of(name) == 'bfloat16'):
            shape = tuple(int(x) for x in field(values[name], 'shape'))
            if (len(shape) == 2 and shape[-1] % PITCH_ELEMENTS and shape[-1] >= MIN_ROW
                    and readers_ok(name)):
                chosen[name] = {'pitch': pitch(shape[-1]), 'shape': list(shape), 'kind': 'product_result',
                                'readers': [output_name(r) for r in readers.get(name, [])],
                                'extra_elements': storage_elements(shape, strides(shape, pitch(shape[-1])))
                                - math.prod(shape)}
            continue
        # The explicit sparse CE writer replaces the original narrowing cast.
        # It stores the same logical BF16 tensor and implements this existing
        # matrix-only pitch contract; no vocabulary/K dimension is enlarged.
        target = str(field(op, 'target', field(op, 'op'))).removeprefix('torch.ops.')
        pitch_aware_ce = target == 'latch_ops.exact_vocab_grad.default'
        if (base not in PRODUCER_BASES and not pitch_aware_ce) or name in protected or name not in values:
            continue
        if dtype_of(name) != 'bfloat16':
            continue
        shape = tuple(int(x) for x in field(values[name], 'shape'))
        misaligned = shape[-1] % (PITCH_ALIGN if PAD_ALIGNED_ROWS else PITCH_ELEMENTS) != 0
        if len(shape) not in (2, 3) or not misaligned or shape[-1] < MIN_ROW:
            continue
        ok, products = matrix_only(name)
        if not ok:
            continue
        chosen[name] = {'pitch': pitch(shape[-1]), 'shape': list(shape), 'kind': 'operand', 'readers': products,
                        'extra_elements': storage_elements(shape, strides(shape, pitch(shape[-1]))) - math.prod(shape)}
    return chosen
