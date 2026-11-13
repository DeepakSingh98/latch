"""Graph -> Program: decide what runs in which phase, and where every value lives.

The kernel is a sequence of phases separated by grid-wide barriers. Inside a
phase every thread block works on every item, so an item may read only what
earlier phases wrote. Lowering works in five steps:

1. Storage. A view becomes a layout over its source and costs nothing. Every
   other operation gets a dense buffer of its own, which is kept only if
   something ends up reading the result from memory.
2. Levels. An operation's level is one more than the latest level that writes
   what it reads, with one exception: a pointwise operation may share a level
   with the pointwise operation of the same shape that feeds it, because both
   then sit in one loop and the value passes in a register. Pointwise operations
   with slack are then moved as late as their readers allow, into those loops.
3. Commit. A new parameter or optimizer tensor is written straight over the old
   one if every other read of the old one is in an earlier phase; otherwise it
   is copied over in a final phase.
4. Items. Pointwise operations of one level and one shape form one loop. Matrix
   products, wide reductions and scatters are items of their own.
5. Arena. Temporaries are packed into one allocation, sharing space whenever
   their lifetimes do not overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from .ir import CompileError, Graph, Op, Ref, Value
from .layout import VIEWS, Layout
from .ops import POINTWISE, REDUCTIONS, SIZE, reduced_extent

WIDE_REDUCTION = 1024  # reduce this many elements per result, or more, with a whole thread block
ALIGN = 64             # arena alignment in bytes
CONSTANTS = {"aten.full.default": 1, "aten.full_like.default": 1, "aten.scalar_tensor.default": 0}  # value argument
BLOCKS = {"aten.mm.default": "gemm", "aten.bmm.default": "gemm"}


@dataclass
class Buffer:
    name: str
    dtype: str
    numel: int
    arg: int | None = None  # position in the kernel's argument table; None means it lives in the arena
    offset: int = 0         # byte offset in the arena

    @property
    def bytes(self) -> int:
        return self.numel * SIZE[self.dtype]


@dataclass(eq=False)
class Loop:
    """One fused pointwise loop over ``shape``; ``stores`` are the results written to memory."""
    level: int
    shape: tuple[int, ...]
    ops: list[Op] = field(default_factory=list)
    stores: list[str] = field(default_factory=list)
    kind = "loop"


@dataclass(eq=False)
class Block:
    """An item with its own parallel structure: "gemm", "reduce" or "scatter"."""
    level: int
    kind: str
    op: Op


@dataclass
class Program:
    graph: Graph
    values: dict[str, Value]
    buffers: dict[str, Buffer]
    layout: dict[str, Layout]    # where each value is, if it is in memory
    local: dict[str, Loop]       # the loop that computes each pointwise value
    constants: dict[str, float]  # values that are one number everywhere; loops use the literal
    phases: list[list]
    arena_bytes: int


def lower(graph: Graph) -> Program:
    return _Lowering(graph).program


class _Lowering:
    def __init__(self, graph: Graph):
        self.graph, self.values = graph, dict(graph.values)
        self.buffers, self.layout, self.constants = {}, {}, {}
        self.ops, self.kind = [], {}  # operations that compute something, in order; name -> kind
        self.producer = {}            # value -> the operation that must run before it is in memory (None: bound)
        for value in self.values.values():
            if value.numel >= 2**31:
                raise CompileError(f"{value.name}: tensors are limited to 2**31 elements")
        self.resolve_storage()
        self.find_readers()
        self.assign_levels()
        self.commit()
        items = self.make_items()
        arena = self.plan_arena(items)
        phases = [[item for item in items if item.level == level] for level in sorted({item.level for item in items})]
        self.program = Program(graph, self.values, self.buffers, self.layout, self.local, self.constants, phases, arena)

    # ------------------------------------------------------------ 1. storage
    def resolve_storage(self):
        for position, name in enumerate(self.graph.inputs):
            self.buffers[name] = Buffer(name, self.values[name].dtype, self.values[name].numel, arg=position)
            self.layout[name], self.producer[name] = Layout.of(name, self.values[name].shape), None
        for op in self.graph.ops:
            self.add(op)
        self.updates = {}
        for target, new in self.graph.updates.items():
            if self.buffers[self.layout[new].buffer].arg is not None:
                new = self.copy(new)  # a view of bound memory: snapshot it while that memory is unchanged
            self.updates[target] = new

    def add(self, op: Op):
        source = op.args[0].name if op.args and isinstance(op.args[0], Ref) else None
        same_dtype = source and self.values[source].dtype == self.values[op.name].dtype
        if op.target in VIEWS or (op.target == "aten._to_copy.default" and same_dtype):
            rule = VIEWS.get(op.target, VIEWS["aten.alias.default"])
            arguments, keywords = [self.values[op.name].shape, *op.args[1:]], op.kwargs if op.target in VIEWS else {}
            layout = rule(self.layout[source], *arguments, **keywords)
            if layout is None:  # not expressible as strides over the source: view a dense copy instead
                source = self.copy(source)
                layout = rule(self.layout[source], *arguments, **keywords)
            self.layout[op.name], self.producer[op.name] = layout, self.producer[source]
        elif op.target == "aten.index_put.default":
            # Copy the base into a buffer, then scatter into that buffer one level later.
            base = self.copy(source)
            self.compute(Op(op.name, op.target, (Ref(base), *op.args[1:]), op.kwargs), "scatter", self.layout[base])
        elif op.target in BLOCKS:
            self.compute(op, BLOCKS[op.target])
        elif op.target in REDUCTIONS and reduced_extent(op, self.values[source].shape) >= WIDE_REDUCTION:
            self.compute(op, "reduce")
        elif op.target in POINTWISE:
            if op.target in CONSTANTS:
                self.constants[op.name] = op.args[CONSTANTS[op.target]]
            self.compute(op, "pointwise")
        else:
            raise CompileError(f"no rule for {op.target} (computing {op.name})")

    def compute(self, op: Op, kind: str, layout: Layout | None = None):
        if layout is None:
            value = self.values[op.name]
            self.buffers[op.name] = Buffer(op.name, value.dtype, value.numel)
            layout = Layout.of(op.name, value.shape)
        self.ops.append(op)
        self.kind[op.name], self.layout[op.name], self.producer[op.name] = kind, layout, op

    def copy(self, name: str, suffix: str = "copy") -> str:
        """A dense copy of ``name`` as a new pointwise value."""
        copy = f"{name}_{suffix}"
        while copy in self.values:
            copy += "_"
        self.values[copy] = Value(copy, self.values[name].shape, self.values[name].dtype)
        self.compute(Op(copy, "latch.copy", (Ref(name),)), "pointwise")
        return copy

    # ------------------------------------------------------------ 2. levels
    def reads(self, op: Op) -> list[tuple[str, int]]:
        """(value, gap) for each tensor ``op`` reads: it must run ``gap`` levels after the value's producer.

        The gap is 0 only from one pointwise operation to another of the same shape:
        at the same level they share a loop and the value never touches memory.
        """
        if self.kind[op.name] != "pointwise":
            return [(name, 1) for name in op.refs()]
        if op.target in CONSTANTS:
            return []
        memory, result = POINTWISE[op.target].memory(op), []
        for name in op.refs():
            if name in memory:
                result.append((name, 1))
            elif name not in self.constants:  # a constant operand is written as a literal
                producer = self.producer[name]
                fused = (producer is not None and producer.name == name and self.kind[name] == "pointwise"
                         and self.values[name].shape == self.values[op.name].shape)
                result.append((name, 0 if fused else 1))
        return result

    def find_readers(self):
        """Keep only operations something needs, and record who reads each one's result."""
        if self.producer[self.graph.loss] is None:
            raise CompileError("the loss must be computed, not passed through")
        needed = {self.producer[name].name for name in [self.graph.loss, *self.updates.values()]}
        self.readers = {}  # producer name -> [(reading op, gap)]
        for op in reversed(self.ops):
            if op.name in needed:
                for name, gap in self.reads(op):
                    if self.producer[name] is not None:
                        needed.add(self.producer[name].name)
                        self.readers.setdefault(self.producer[name].name, []).append((op, gap))
        self.ops = [op for op in self.ops if op.name in needed]

    def assign_levels(self):
        self.level = {}
        for op in self.ops:  # as early as possible
            self.level[op.name] = max((self.level[self.producer[name].name] + gap for name, gap in self.reads(op)
                                       if self.producer[name] is not None), default=0)
        for op in reversed(self.ops):  # then pointwise work as late as its readers allow, to join their loops
            if self.kind[op.name] == "pointwise" and op.name in self.readers:
                self.level[op.name] = min(self.level[reader.name] - gap for reader, gap in self.readers[op.name])

    # ------------------------------------------------------------ 3. commit
    def commit(self):
        final = max(self.level.values(), default=-1) + 1
        old = {}  # buffer -> [(reading op, whether it reads element i in a loop over the buffer's own dense shape)]
        for op in self.ops:
            memory = POINTWISE[op.target].memory(op) if self.kind[op.name] == "pointwise" else op.refs()
            for name, _ in self.reads(op):
                layout = self.layout[name]
                in_step = (name not in memory and layout.dense and layout.offset == 0
                           and layout.shape == self.values[op.name].shape == self.values[layout.buffer].shape)
                old.setdefault(layout.buffer, []).append((op, in_step))
        for target, new in self.updates.items():
            producer = self.producer[new]
            if producer.name == new and self.kind[new] == "pointwise" and self.buffers[new].arg is None and all(
                    self.level[op.name] < self.level[new] or (self.level[op.name] == self.level[new] and in_step)
                    for op, in_step in old.get(target, ())):
                # Every other read of the old tensor is earlier, or is the same element in the same loop:
                # the new tensor's buffer can be the bound tensor itself.
                self.buffers[new] = replace(self.buffers[target], name=new)
            else:
                copy = self.copy(new, "commit")
                self.buffers[copy], self.level[copy] = replace(self.buffers[target], name=copy), final
                self.readers.setdefault(producer.name, []).append((self.ops[-1], 1))

    # ------------------------------------------------------------ 4. items
    def make_items(self) -> list:
        loops, items, self.local = {}, [], {}
        kept = {self.producer[self.graph.loss].name, *(b.name for b in self.buffers.values() if b.arg is not None)}
        for op in self.ops:
            level = self.level[op.name]
            if self.kind[op.name] != "pointwise":
                items.append(Block(level, self.kind[op.name], op))
                continue
            shape = self.values[op.name].shape
            if (level, shape) not in loops:
                loops[level, shape] = Loop(level, shape)
                items.append(loops[level, shape])
            loop = self.local[op.name] = loops[level, shape]
            loop.ops.append(op)
            # In memory if it is an output, or if anything reads it from outside this loop.
            readers = self.readers.get(op.name, ())
            if op.name in kept or any(gap or self.level[reader.name] != level for reader, gap in readers):
                loop.stores.append(op.name)
        return items

    # ------------------------------------------------------------ 5. arena
    def touched(self, item) -> set[str]:
        """Buffers an item reads from or writes to memory."""
        if item.kind != "loop":
            return {self.layout[name].buffer for name in [item.op.name, *item.op.refs()]}
        names = set(item.stores)
        for op in item.ops:
            names.update(name for name, _ in self.reads(op) if self.local.get(name) is not item)
        return {self.layout[name].buffer for name in names}

    def plan_arena(self, items) -> int:
        """Place every temporary; two may overlap only if their live ranges of levels are disjoint."""
        first, last = {}, {}
        for item in items:
            for name in self.touched(item):
                if self.buffers[name].arg is None:
                    first[name] = min(first.get(name, item.level), item.level)
                    last[name] = max(last.get(name, item.level), item.level)
        last[self.layout[self.graph.loss].buffer] = math.inf  # read by the host after the kernel returns
        placed, total = [], 0
        for name in sorted(first, key=lambda name: (first[name], -self.buffers[name].bytes)):
            size, offset = -(-self.buffers[name].bytes // ALIGN) * ALIGN, 0
            for start, end in sorted((o, o + s) for o, s, f, l in placed if not (l < first[name] or f > last[name])):
                if offset + size <= start:
                    break
                offset = max(offset, end)
            placed.append((offset, size, first[name], last[name]))
            self.buffers[name].offset, total = offset, max(total, offset + size)
        return total
