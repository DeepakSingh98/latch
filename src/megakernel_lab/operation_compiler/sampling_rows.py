"""Row-task execution of a nondifferentiated sampling loop (C-OPS4-B).

Reuses the partition rule of the S100L/S101E/S103H energy-model generators (research/s101/S101E/SPEC.md R1,
src/s103h_ebm/compiler.py H1 at 106d37ced): the rows of the sampling carry are split into tasks of RG rows, each
task runs on its own GROUP of CTAs for ALL loop iterations, and groups synchronize only among themselves (the
old engines' group flag protocol); the whole grid meets again only after the loop. Here the rule is derived
from graph properties instead of a model declaration:

  * row separability: every value of the step region either carries the row axis of the carry (dimension 0 of
    size B), or is row-free (loop-invariant parameters/casts, scalars, constants); products contract only
    row-free right operands, reductions never combine rows except into values used solely for their shape
    (PyTorch's ones_like seed of grad(sum)); counter-grid random fields address absolute rows
  * the region is re-instantiated at RG rows (shape substitution on the captured IR) and lowered by the same
    operation lowering; random fields receive the task's row base
  * the generated region is executed on a virtual grid: its grid-strided loops and product tile slices are
    remapped to the group's CTA range and its barriers become group barriers (one release/acquire arrival
    counter per group); engines without a virtual-grid interface refuse the rule with a recorded reason
  * each group owns a private copy of the region scratch; step-changed state is refused (only the carry rows
    are double buffered)

Nothing about the model is inspected: the same rule applies to any sampling/forward loop whose step is
row-separable (e.g. per-example inner iterations), and refuses otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

from .ir import Operation, Ref, Value
from .lower import LoweringError, _ELEMENTWISE_BASES, _contiguous
from .loop import OperationIR

FIELD = "latch_ops.philox_field.default"
SHAPE_ONLY = {"ones_like", "zeros_like", "empty_like", "full_like", "new_zeros", "new_ones", "new_full", "new_empty"}
VIEWS = {"detach", "alias", "view", "_unsafe_view", "reshape", "unsqueeze", "squeeze", "clone", "contiguous",
         "t", "transpose", "permute", "expand", "select", "slice"}
REDUCED = "reduced"


class RowRefusal(ValueError):
    pass


def _base(target):
    return target.removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]


def _refs(obj):
    if isinstance(obj, Ref):
        yield obj.name
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _refs(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _refs(item)


def row_axes(region, carry_inputs, B):
    """Value -> row axis (int), None (row-free) or REDUCED; RowRefusal names the first non-separable op."""
    values = region.values
    axes = {}
    for name in region.inputs:
        shape = tuple(values[name].shape)
        if name in carry_inputs:
            if not shape or shape[0] != B:
                raise RowRefusal(f"carry input {name} has no leading row extent {B}")
            axes[name] = 0
        else:
            axes[name] = None
    for op in region.operations:
        out = op.outputs[0] if op.outputs else op.name
        shape = tuple(values[out].shape)
        base = _base(op.target)
        reads = [r for r in dict.fromkeys(_refs((op.args, op.kwargs)))]
        ins = {r: axes[r] for r in reads}

        def aligned(r):
            a = ins[r]
            return None if a is None or a == REDUCED else a + (len(shape) - len(values[r].shape))

        if any(a == REDUCED for a in ins.values()) and base not in SHAPE_ONLY:
            raise RowRefusal(f"{op.name} ({op.target}) reads a value reduced over rows")
        if op.target == FIELD:
            axes[out] = 0 if shape[0] == B else None
            continue
        if base in SHAPE_ONLY:
            template = reads[0] if reads else None
            t_axis = ins.get(template)
            axes[out] = None if t_axis in (None, REDUCED) else t_axis
            continue
        if base in ("mm", "matmul"):
            a, b = reads[0], reads[1]
            if ins[b] is not None:
                raise RowRefusal(f"{op.name}: product contracts or broadcasts the row axis of its right operand")
            if ins[a] not in (None, 0):
                raise RowRefusal(f"{op.name}: row axis is not the product's row dimension")
            axes[out] = ins[a]
            continue
        if base in ("sum", "mean", "amax"):
            source = reads[0]
            a = ins[source]
            rank = len(values[source].shape)
            dims = op.args[1] if len(op.args) > 1 and op.args[1] is not None else list(range(rank))
            dims = [dims] if isinstance(dims, int) else list(dims)
            dims = sorted({int(d) % rank for d in dims}) if dims else list(range(rank))
            keep = bool(op.args[2] if len(op.args) > 2 else op.kwargs.get("keepdim", False))
            if a is None:
                axes[out] = None
            elif a in dims:
                axes[out] = REDUCED
            else:
                axes[out] = a if keep else a - sum(1 for d in dims if d < a)
            continue
        if base in ("t", "transpose", "permute"):
            source = reads[0]
            a = ins[source]
            rank = len(values[source].shape)
            if a is None:
                axes[out] = None
                continue
            if base == "t":
                order = list(reversed(range(rank)))
            elif base == "transpose":
                order = list(range(rank))
                i, j = int(op.args[1]) % rank, int(op.args[2]) % rank
                order[i], order[j] = order[j], order[i]
            else:
                order = [int(d) % rank for d in op.args[1]]
            axes[out] = order.index(a)
            continue
        if base == "expand":
            source = reads[0]
            a = aligned(source)
            if a is not None:
                axes[out] = a
                continue
            padded = (1,) * (len(shape) - len(values[source].shape)) + tuple(values[source].shape)
            grown = [d for d, (old, new) in enumerate(zip(padded, shape)) if new == B and old != B]
            if len(grown) > 1:
                raise RowRefusal(f"{op.name}: ambiguous row broadcast")
            axes[out] = grown[0] if grown else None
            continue
        if base in ("unsqueeze", "squeeze", "view", "_unsafe_view", "reshape", "detach", "alias", "clone",
                    "contiguous", "_to_copy", "lift_fresh_copy"):
            source = reads[0]
            a = ins[source]
            if a is None:
                axes[out] = None
                continue
            old = tuple(values[source].shape)
            prefix = math.prod(old[:a])
            found = [p for p in range(len(shape)) if shape[p] == B and math.prod(shape[:p]) == prefix
                     and math.prod(shape[p + 1:]) == math.prod(old[a + 1:])]
            if len(found) != 1:
                raise RowRefusal(f"{op.name}: view merges or splits the row axis")
            axes[out] = found[0]
            continue
        if base in ("select", "slice"):
            source = reads[0]
            a = ins[source]
            dim = int(op.args[1]) % len(values[source].shape)
            if a is None:
                axes[out] = None
            elif dim == a:
                raise RowRefusal(f"{op.name}: {base} along the row axis")
            else:
                axes[out] = a - (1 if base == "select" and dim < a else 0)
            continue
        if base == "cat":
            dim = int(op.args[1] if len(op.args) > 1 else op.kwargs.get("dim", 0)) % len(shape)
            found = {aligned(r) for r in reads} - {None}
            if dim in found:
                raise RowRefusal(f"{op.name}: cat along the row axis")
            if len(found) > 1:
                raise RowRefusal(f"{op.name}: operands disagree on the row axis")
            axes[out] = found.pop() if found else None
            continue
        if base in _ELEMENTWISE_BASES or base in ("where", "silu_backward", "sigmoid_backward", "threshold_backward"):
            found = {aligned(r) for r in reads} - {None}
            if len(found) > 1:
                raise RowRefusal(f"{op.name}: operands disagree on the row axis")
            axis = found.pop() if found else None
            if axis is not None:
                for r in reads:
                    if ins[r] is None:
                        rshape = (1,) * (len(shape) - len(values[r].shape)) + tuple(values[r].shape)
                        if rshape[axis] not in (1,):
                            raise RowRefusal(f"{op.name}: row-free operand {r} spans the row axis")
            axes[out] = axis
            continue
        if all(a is None for a in ins.values()):
            axes[out] = None
            continue
        raise RowRefusal(f"{op.name}: {op.target} has no row-separability rule")
    for name in region.outputs:
        if axes.get(name) != 0:
            raise RowRefusal(f"region output {name} is not row-indexed along dimension 0")
    return axes


def _patch(obj, old, new):
    if isinstance(obj, (list, tuple)) and len(obj) == len(old) and all(type(x) is int for x in obj) \
            and tuple(obj) == tuple(old):
        return type(obj)(new)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_patch(x, old, new) for x in obj)
    if isinstance(obj, dict):
        return {k: _patch(v, old, new) for k, v in obj.items()}
    return obj


def slice_region(region, axes, B, RG):
    """The region re-instantiated at RG rows, with an appended int64 'row_base' input for random fields."""
    values = {}
    for name, value in region.values.items():
        a = axes.get(name)
        if a is None or a == REDUCED:
            values[name] = replace(value)
            continue
        shape = list(value.shape)
        shape[a] = RG
        strides = tuple(value.strides) if name in region.inputs else tuple(_contiguous(tuple(shape)))
        values[name] = replace(value, shape=tuple(shape), strides=strides)
    base_name = "row_base"
    while base_name in values:
        base_name = "_" + base_name
    values[base_name] = Value(base_name, (), "int64", (), role="input")
    operations = []
    for op in region.operations:
        out = op.outputs[0] if op.outputs else op.name
        args, kwargs = op.args, op.kwargs
        if axes.get(out) not in (None, REDUCED):
            old, new = tuple(region.values[out].shape), tuple(values[out].shape)
            args, kwargs = _patch(args, old, new), _patch(kwargs, old, new)
        if op.target == FIELD and axes.get(out) == 0:
            args = (*args, Ref(base_name))
            reads = (*op.reads, base_name)
            operations.append(replace(op, args=args, kwargs=kwargs, reads=reads, inputs=(*op.inputs, base_name)))
            continue
        operations.append(replace(op, args=args, kwargs=kwargs))
    metadata = dict(region.metadata, row_tasks={"rows": RG, "of": B, "row_base_input": base_name})
    return OperationIR(values, operations, (*region.inputs, base_name), tuple(region.outputs), metadata)


GROUP_BARRIER = r'''
#ifndef LATCH_OPS_GROUP_BARRIER
#define LATCH_OPS_GROUP_BARRIER
// C-OPS4-B row tasks: one release/acquire arrival counter per CTA group (S100L/S101E group flag protocol,
// expressed with the S102G/bar2 counter barrier). The target is monotonic per CTA for the whole loop.
__shared__ unsigned long long latch_group_target;
__device__ __forceinline__ void latch_group_barrier(unsigned long long* counter, unsigned members) {
  __syncthreads();
  if (threadIdx.x == 0) {
    latch_group_target += members;
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(counter) : "memory");
    while (s101h::ld_acq_u64(counter) < latch_group_target) {}
  }
  __syncthreads();
}
#endif
'''

_ALLOWED_CALLS = re.compile(r"\b(popsc::BArgs|popsc::bmm_\w*|s101h::warp_sum|s101h::ld_acq_u64|latch_\w+|"
                            r"__\w+|sqrtf|expf|logf|fminf|fmaxf|fabsf|tanhf|powf|isnan|rsqrtf|expm1f|cospif|sinpif|"
                            r"static_cast|reinterpret_cast|stamp|max|min)\b")


def virtualize(text):
    """Rewrite a lowered region's generated code to run on the group's CTA range; refuse unknown grid uses."""
    problems = []
    for token in ("this_grid().sync", "latch_ctr", "atomicAdd", "atomicInc", "/*B_VBASE*/", "/*VBASE*/",
                  "/*C_VBASE*/", "s101h::gemm_phase", "popsb", "pops2::", "ticket"):
        if token in text:
            problems.append(f"grid-level construct {token!r}")
    for match in re.finditer(r"\b([A-Za-z_]\w*)::(\w+)\s*(<[^;{}()]*>)?\s*\(", text):
        call = f"{match.group(1)}::{match.group(2)}"
        if not _ALLOWED_CALLS.fullmatch(call) and match.group(1) not in ("cg", "popsc"):
            problems.append(f"engine call {call} has no virtual-grid interface")
    slices = []

    def popsc(match):
        lo, hi, den = match.group(2), match.group(3), match.group(4)
        if not all(re.fullmatch(r"\d+", x) for x in (lo, hi, den)):
            problems.append(f"popsc call with runtime slice ({lo}, {hi}, {den})")
            return match.group(0)
        slices.append((int(lo), int(hi), int(den)))
        L, H, D = int(lo), int(hi), int(den)
        return (f"{match.group(1)}(int)(latch_b0 + (long long)lg_ * {L} / {D}), "
                f"(int)(latch_b0 + (long long)lg_ * {H} / {D}), (int)latch_phys_grid, ")
    text = re.sub(r"(popsc::bmm_\w*<[^>]*>\(\s*g\s*,\s*smem\s*,\s*)(\S+?)\s*,\s*(\S+?)\s*,\s*(\S+?)\s*,\s*", popsc, text)
    text = text.replace("latch_loop_bar_begin(latch_bar);", "")
    text = text.replace("latch_loop_barrier(latch_bar);", "latch_group_barrier(latch_gbar, lg_);")
    text = text.replace("grid.sync();", "latch_group_barrier(latch_gbar, lg_);")
    text = re.sub(r"\bblockIdx\.x\b", "lb_", text)
    text = re.sub(r"\bgridDim\.x\b", "lg_", text)
    return text, problems, slices


@dataclass
class RowSchedule:
    manifest: dict
    functions: str
    helpers: str
    shared_bytes: int
    program: object
    tasks: int
    rows: int
    scratch_words: int

    def loop_code(self, *, pw, first, boundary, state_pointers, data, ktable, scratch_offset, counter_offset,
                  base_offset, carry_row_bytes, nc):
        G, RG = self.tasks, self.rows
        def offset(expression, c):
            return f'reinterpret_cast<float*>(reinterpret_cast<char*>({expression}) + (long long)latch_grp * {RG * carry_row_bytes[c]}LL)'
        cur = [f'((iteration) % 2 == 0 ? {a} : {b})' for a, b in zip(*boundary)]
        nxt = [f'((iteration + 1) % 2 == 0 ? {a} : {b})' for a, b in zip(*boundary)]
        args = (pw + [offset(cur[c], c) for c in range(nc)] + state_pointers(cur)
                + [f'reinterpret_cast<float*>(reinterpret_cast<int64_t*>(arena + {ktable}LL) + iteration)'] + data
                + [f'reinterpret_cast<float*>(reinterpret_cast<int64_t*>(arena + {base_offset}LL) + latch_grp)'])
        outs = [offset(nxt[c], c) for c in range(nc)]
        return '\n'.join([
            '{',
            '  // C-OPS4-B row tasks: group g owns CTAs [G g / T, G (g+1) / T) and carry rows [g RG, (g+1) RG).',
            '  const int latch_phys_grid = (int)gridDim.x;',
            f'  int latch_grp = (int)(((long long)blockIdx.x * {G} + {G} - 1) / latch_phys_grid);',
            f'  while (latch_grp > 0 && (int)((long long)latch_phys_grid * latch_grp / {G}) > (int)blockIdx.x) --latch_grp;',
            f'  while (latch_grp + 1 < {G} && (int)((long long)latch_phys_grid * (latch_grp + 1) / {G}) <= (int)blockIdx.x) ++latch_grp;',
            f'  const unsigned latch_b0 = (unsigned)((long long)latch_phys_grid * latch_grp / {G});',
            f'  const unsigned lg_ = (unsigned)((long long)latch_phys_grid * (latch_grp + 1) / {G}) - latch_b0;',
            '  const unsigned lb_ = blockIdx.x - latch_b0;',
            f'  unsigned long long* latch_gbar = reinterpret_cast<unsigned long long*>(arena + {counter_offset}LL) + 8 * latch_grp;',
            f'  float* latch_scratch = arena + {scratch_offset}LL + (long long)latch_grp * {self.scratch_words}LL;',
            '  for (int iteration = 0; iteration < count; ++iteration) {',
            '    float* region_args[] = {' + ', '.join(args) + '};',
            '    float* region_outputs[] = {' + ', '.join(outs) + '};',
            '    sampled_rows_body(region_args, latch_scratch, region_outputs, lb_, lg_, latch_b0, latch_phys_grid, latch_gbar'
            + (', (ticks && iteration == 0 && latch_grp == 0) ? ticks + 8 : nullptr' if getattr(self, 'ticks', False) else '')
            + ');',
            '  }',
            '}',
            'grid.sync();'])

    def prologue(self, counter_offset, base_offset):
        G, RG = self.tasks, self.rows
        return '\n'.join([
            'if (threadIdx.x == 0) latch_group_target = 0;',
            f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {8 * G}LL; '
            f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<unsigned long long*>(arena + {counter_offset}LL)[i] = 0ULL;',
            f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {G}LL; '
            f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<int64_t*>(arena + {base_offset}LL)[i] = i * {RG}LL;'])


def plan_row_tasks(graph, body_program, chosen, region_options):
    """RowSchedule for the sampling body, or None with the refusal recorded in chosen['row_refusal']."""
    from .lower import lower
    from .sampling_lower import region_function
    m = graph.metadata
    body = graph.body
    n = len(m["parameter_names"]) + len(m.get("invariant_casts", []))
    nc = len(m["carry_names"])
    carry_inputs = set(body.inputs[n:n + nc])
    reasons = []
    if m.get("body_state_changes"):
        reasons.append("the step changes explicit state; only carry rows are partitioned")
    B = None
    if carry_inputs:
        B = body.values[body.inputs[n]].shape[0] if body.values[body.inputs[n]].shape else None
    options = dict(chosen.get("row_options") or {})
    rows = options.get("rows", "auto")
    if rows == "auto" and B:
        # Group rule: tasks of 64 rows (the S103H WB/SD plan size) when the batch allows at least two tasks,
        # otherwise of 32 rows; never fewer than 32 rows (BF16 product tiles are 32-row multiples).
        rows = 64 if B >= 128 and B % 64 == 0 else 32 if B >= 64 and B % 32 == 0 else None
    if not B or not rows or B % rows or B // rows < 2:
        reasons.append(f"carry rows {B} do not split into at least two tasks of {rows} rows")
    axes = None
    if not reasons:
        try:
            axes = row_axes(body, carry_inputs, B)
        except RowRefusal as exc:
            reasons.append(f"step region is not row separable: {exc}")
    if reasons:
        if chosen["row_tasks"] == "on":
            raise LoweringError("row tasks refused: " + "; ".join(reasons))
        chosen["row_refusal"] = reasons
        return None
    sliced = slice_region(body, axes, B, rows)
    program = lower(sliced, **region_options, **(chosen.get("body_options") or {}))
    if program.manifest["state_writes"]:
        raise LoweringError("row-task region must be pure")
    ticks = bool(chosen.get("phase_ticks"))
    function = region_function("sampled", "rows_body", program, ticks=ticks)
    text, problems, slices = virtualize(function)
    head = "void sampled_rows_body(float** args, float* arena, float** out" + (", unsigned long long* ticks)" if ticks else ")")
    text = text.replace(head, "void sampled_rows_body(float** args, float* arena, float** out, unsigned lb_, unsigned lg_, "
                              "unsigned latch_b0, int latch_phys_grid, unsigned long long* latch_gbar"
                              + (", unsigned long long* ticks)" if ticks else ")"))
    if program.functions:
        problems.append("namespaced region helper functions are not virtualized")
    if problems:
        if chosen["row_tasks"] == "on":
            raise LoweringError("row tasks refused: " + "; ".join(sorted(set(problems))))
        chosen["row_refusal"] = sorted(set(problems))
        return None
    tasks = B // rows
    manifest = {"selected": True, "rule": "S101E/S103H row tasks on CTA groups; group barriers; private scratch",
                "rows_per_task": rows, "tasks": tasks, "row_extent": B,
                "region_phases": len(program.manifest.get("phases", [])),
                "popsc_slices_rewritten": len(slices),
                "per_group_scratch_bytes": 4 * program.manifest["arena_words"],
                "region_manifest": program.manifest}
    schedule = RowSchedule(manifest, text, GROUP_BARRIER + ("\n" + program.helpers if program.helpers else ""),
                           program.manifest.get("shared_bytes", 17408), program, tasks, rows,
                           (program.manifest["arena_words"] + 63) // 64 * 64)
    schedule.ticks = ticks
    return schedule
