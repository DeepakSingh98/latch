"""Program -> the source of one kernel.

The kernel body is the phases in order with a grid barrier between them. Every
item spreads its own work over all threads of all blocks: thread ``tid`` of
``threads`` takes elements tid, tid + threads, ... of a loop.
"""
from __future__ import annotations

import math

from .ir import CompileError, Ref
from .layout import row_major
from .lower import Program
from .ops import CTYPE, POINTWISE, REDUCTIONS, literal, reduced

GEMM_TASKS = 64  # prefer the largest matrix tile that still leaves this many tiles to share out


def generate(program: Program) -> str:
    tile = max([sum(_tile(program, item.op)) for phase in program.phases for item in phase if item.kind == "gemm"],
               default=0)
    lines = ['#include "latch.cuh"', f"#define LATCH_SHARED_BYTES latch::Shared<{tile}>::bytes", "",
             f"// {sum(len(phase) for phase in program.phases)} items in {len(program.phases)} phases; "
             f"arena {program.arena_bytes} bytes",
             "LATCH_KERNEL void latch_step(void** args, char* arena) {",
             "  LATCH_SHARED(shared, LATCH_SHARED_BYTES);",
             "  const int tid = blockIdx.x * latch::THREADS + threadIdx.x, threads = gridDim.x * latch::THREADS;"]
    for number, phase in enumerate(program.phases):
        lines.append(f"  // ---- phase {number}")
        for item in phase:
            lines += ["  " + line for line in _Item(program, item).text().splitlines()]
        if number + 1 < len(program.phases):
            lines.append("  LATCH_GRID_SYNC();")
    lines += ["}", "LATCH_EXPORTS(latch_step)", ""]
    return "\n".join(lines)


def _tile(program: Program, op) -> tuple[int, int]:
    """(TM, TN): each thread's patch of the output tile, 1, 2 or 4 on a side."""
    shape = program.values[op.name].shape
    for size in (4, 2):
        tiles = math.ceil(shape[-2] / (16 * size)) * math.ceil(shape[-1] / (16 * size))
        if tiles * math.prod(shape[:-2]) >= GEMM_TASKS:
            return size, size
    return 1, 1


def _unravel(index: str, extents) -> list[str]:
    """C expressions for the coordinates of row-major position ``index`` in a box of ``extents``."""
    coords = []
    for extent, step in zip(extents, row_major(extents)):
        text = index if step == 1 else f"{index} / {step}"
        coords.append(text if step * extent == math.prod(extents) else f"{text} % {extent}")
    return [f"({c})" for c in coords]


class _Item:
    """Emits one item. Also the ``e`` that pointwise rules in ops.py write against."""

    def __init__(self, program: Program, item):
        self.p, self.item, self.pointers, self.used_coords, self.pending = program, item, {}, set(), []
        self.out = None

    def text(self) -> str:
        body = getattr(self, self.item.kind)()
        declarations = [f"{CTYPE[b.dtype]}* const b_{b.name} = ({CTYPE[b.dtype]}*)"
                        + (f"args[{b.arg}];" if b.arg is not None else f"(arena + {b.offset});")
                        for b in self.pointers.values()]
        return "\n".join(["{", *("  " + line for line in declarations + body), "}"])

    # ------------------------------------------------------------ what rules use
    def dtype(self, ref) -> str:
        return self.p.values[ref.name].dtype

    def shape(self, ref) -> tuple[int, ...]:
        return self.p.values[ref.name].shape

    def pointer(self, buffer: str) -> str:
        self.pointers[buffer] = self.p.buffers[buffer]
        return f"b_{buffer}"

    def coord(self, d: int) -> str:
        """The loop's coordinate along dimension ``d`` of its shape."""
        self.used_coords.add(d)
        return f"c{d}"

    def before(self, statement: str):
        self.pending.append(statement)

    def arg(self, x, dtype=None) -> str:
        """An operand at the loop's position, converted to ``dtype`` (default: the result's)."""
        dtype = dtype or self.out.dtype
        if not isinstance(x, Ref):
            return literal(x, dtype)
        if x.name in self.p.constants:
            return literal(self.p.constants[x.name], dtype)
        if self.p.local.get(x.name) is self.item:
            text = f"x_{x.name}"
        else:
            layout, shape = self.p.layout[x.name], self.item.shape
            if layout.shape == shape and layout.dense:
                text = f"{self.pointer(layout.buffer)}[{layout.offset} + i]" if layout.offset else \
                    f"{self.pointer(layout.buffer)}[i]"
            else:  # broadcast: dimensions align at the right, and extent-1 dimensions stay at 0
                lead = len(shape) - len(layout.shape)
                text = self.at(x, [self.coord(lead + d) if n != 1 else "0" for d, n in enumerate(layout.shape)])
        return text if self.dtype(x) == dtype else f"(({CTYPE[dtype]}){text})"

    def at(self, ref, coords, layout=None) -> str:
        """The element of a value in memory at explicit coordinates."""
        layout = layout or self.p.layout[ref.name]
        terms = [str(layout.offset)] if layout.offset else []
        terms += [f"{s} * {c}" for n, s, c in zip(layout.shape, layout.strides, coords) if n != 1 and s != 0]
        return f"{self.pointer(layout.buffer)}[{' + '.join(terms) or '0'}]"

    # ------------------------------------------------------------ items
    def loop(self) -> list[str]:
        loop, body = self.item, []
        for op in loop.ops:
            self.out = self.p.values[op.name]
            expression = POINTWISE[op.target].emit(self, op)
            body += self.pending + [f"const {CTYPE[self.out.dtype]} x_{op.name} = {expression};"]
            self.pending = []
        for name in loop.stores:  # after every read, so a result may overwrite an operand of the same loop
            layout = self.p.layout[name]
            body.append(f"{self.pointer(layout.buffer)}[i] = x_{name};")
        coords = _unravel("i", loop.shape)
        head = [f"const int c{d} = {coords[d]};" for d in sorted(self.used_coords)]
        count = math.prod(loop.shape)
        return [f"// loop over {list(loop.shape)}: {' '.join(loop.stores)}",
                f"for (int i = tid; i < {count}; i += threads) {{", *("  " + line for line in head + body), "}"]

    def gemm(self) -> list[str]:
        op = self.item.op
        a, b = (self.p.layout[ref.name] for ref in op.args[:2])
        batch = a.shape[0] if len(a.shape) == 3 else 1
        (sab, sam, sak), (sbb, sbk, sbn) = (((0,) + l.strides)[-3:] for l in (a, b))
        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        tm, tn = _tile(self.p, op)
        depth = next(d for d in (4, 8, 16, 32, 64) if d >= min(k, 64))  # staged per pass; a constant so loops unroll
        return [f"// {op.name} = {op.args[0].name} @ {op.args[1].name}: {batch} x [{m}, {k}] @ [{k}, {n}]",
                f"latch::gemm<{tm}, {tn}, {depth}>({self.pointer(a.buffer)} + {a.offset}, {sab}, {sam}, {sak}, "
                f"{self.pointer(b.buffer)} + {b.offset}, {sbb}, {sbk}, {sbn}, "
                f"{self.pointer(self.p.layout[op.name].buffer)}, {batch}, {m}, {k}, {n}, shared);"]

    def reduce(self) -> list[str]:
        """A wide reduction: one block per result, its threads sharing out the reduced elements."""
        op, spec = self.item.op, REDUCTIONS[self.item.op.target]
        shape = self.shape(op.args[0])
        dims, _ = reduced(op, shape)
        kept = [d for d in range(len(shape)) if d not in dims]
        coords = dict(zip(kept, _unravel("o", [shape[d] for d in kept])))
        coords.update(zip(dims, _unravel("r", [shape[d] for d in dims])))
        results, count = math.prod(shape[d] for d in kept), math.prod(shape[d] for d in dims)
        ctype = CTYPE[self.p.values[op.name].dtype]
        element = f"({ctype}){self.at(op.args[0], [coords[d] for d in range(len(shape))])}"
        result = f"acc / {literal(count, 'float32')}" if spec.mean else "acc"
        return [f"// {op.name} = {op.target.split('.')[1]} of {op.args[0].name}: {count} elements for each of {results}",
                f"for (int o = blockIdx.x; o < {results}; o += gridDim.x) {{",
                f"  {ctype} acc = {spec.initial(self.p.values[op.name].dtype)};",
                f"  for (int r = threadIdx.x; r < {count}; r += latch::THREADS) acc = {spec.combine('acc', element)};",
                f"  acc = latch::block_reduce(acc, shared, []({ctype} a, {ctype} b) {{ return {spec.combine('a', 'b')}; }});",
                f"  if (threadIdx.x == 0) {self.pointer(self.p.layout[op.name].buffer)}[o] = {result};",
                "}"]

    def scatter(self) -> list[str]:
        """index_put on the leading dimension: result[index[...], ...] (+)= values."""
        op = self.item.op
        indices, values = op.args[1], op.args[2]
        accumulate = op.args[3] if len(op.args) > 3 else op.kwargs.get("accumulate", False)
        out = self.p.layout[op.name]
        if len(indices) != 1 or indices[0] is None or self.p.values[op.name].dtype != "float32":
            raise CompileError(f"{op.name}: index_put needs one index tensor on the first dimension of a float tensor")
        index = self.p.layout[indices[0].name]
        full = index.shape + out.shape[1:]  # what the values broadcast to
        coords, lead = _unravel("i", full), len(index.shape)
        row = self.at(indices[0], coords[:lead])
        inner = [f"{s} * {c}" for s, c in zip(out.strides[1:], coords[lead:])]
        target = f"{self.pointer(out.buffer)}[{' + '.join([f'{out.strides[0]} * {row}'] + inner)}]"
        source = self.p.layout[values.name]
        value = self.at(values, [c if n != 1 else "0" for c, n in zip(coords[len(full) - len(source.shape):], source.shape)])
        store = f"latch::atomic_add(&{target}, {value});" if accumulate else f"{target} = {value};"
        return [f"// {op.name}: scatter {values.name} into rows chosen by {indices[0].name}",
                f"for (int i = tid; i < {math.prod(full)}; i += threads) {store}"]
