"""Reusable packed attention lowering; dynamic lengths never enter host control."""
from __future__ import annotations

import math
from .segmented_attention import BACKWARD, CONTRACT, FORWARD, TARGETS, FORWARD_WORK, BACKWARD_WORK, NATIVE_WIDTHS

HEADER = "attention_segmented.cuh"
HEADERS = (HEADER, "attention_long_flat.cuh")
# Width-32 heads use schedule 6 of attention_segmented.cuh by default: 128-row forward/dQ items (two overlapped row
# blocks per warpgroup) and 128-key dK/dV items, with the FTZ ex2 exponential (contract suffix "+ftz-ex2"; differs
# from exp2f only for probabilities below 2^-126). The selection is recorded in every phase's metadata and source.
DEFAULT_W32_VARIANT = 6
# Opt-in width-32 forms (ops.segmented_attention_rules), both on schedule 8's two-warpgroup region:
#   folded_forward: forward form 9 -- schedule 6's forward with the softmax scale folded into the exponent and Q
#     tiles read by the warpgroup MMA from shared memory (template int 8 + 16 * 9 = 152).
#   fused_backward: backward form 10 -- single-pass dK/dV items that also form dS and add shared-memory-staged dQ
#     tiles to a [heads, rows, 32] FP32 accumulator by asynchronous bulk reductions, taking tickets in
#     segment-major order (template int 8 + 16 * 10 = 168); a final pass converts the accumulator to BF16 dQ.
FORWARD_FORM = 9
BACKWARD_FORM = 10


def _rule(name):
    from .segmented_attention_rules import active
    return active(name)


def native_body(width, variant):
    return "pflatlong_v7_bf16_packed" if width == 64 else f"packed_w32_v{variant}_bf16"


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define, target_arch):
    from .lower import LoweringError
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ()))
    target = str(field(op, "target"))
    fwd = target == FORWARD
    if target not in TARGETS or len(args) != (6 if fwd else 9) or len(outputs) != (3 if fwd else 4):
        raise LoweringError("invalid packed segmented attention ABI")
    if target_arch != "sm_90":
        raise LoweringError("packed segmented attention currently requires native sm_90 WGMMA")
    tensor_args = args[:-2]
    ins = [ref(a, names) for a in tensor_args]
    q, k, v = ins[:3]
    offsets = ins[-1]
    bound, scale = args[-2:]
    shape = tuple(layouts[q].shape)
    if (len(shape) != 3 or shape[-1] not in NATIVE_WIDTHS or shape[1] < 1 or type(bound) is not int
            or not 1 <= bound <= 4096 or not math.isfinite(scale) or scale <= 0):
        raise LoweringError("packed attention requires width 32 or 64, positive scale and max_seqlen in 1..4096")
    for name in (q, k, v, *([] if fwd else [ins[3], ins[5]])):
        if layouts[name].shape != shape or not layouts[name].contiguous or dtypes[name] != "bfloat16":
            raise LoweringError("packed attention operands must be matching contiguous BF16 tensors")
    rows, heads, width = shape
    variant = DEFAULT_W32_VARIANT if width == 32 else 0
    fused = width == 32 and not fwd and _rule("fused_backward")
    lean_forward = width == 32 and fwd and _rule("folded_forward")
    if fused or lean_forward:
        variant = 8
    body = native_body(width, variant)
    lo = layouts[offsets]
    if len(lo.shape) != 1 or lo.shape[0] < 2 or not lo.contiguous or dtypes[offsets] not in ("int32", "int64"):
        raise LoweringError("packed attention offsets must be contiguous int32/int64 [segments+1]")
    groups = lo.shape[0] - 1
    if not fwd and (layouts[ins[4]].shape != (heads, rows) or not layouts[ins[4]].contiguous
                    or dtypes[ins[4]] != "float32"):
        raise LoweringError("packed attention statistics must be contiguous FP32 [heads, rows]")
    for name in outputs:
        define(name, tuple(field(values[name], "shape")), None, dtypes[name])
    prefix = outputs[0] + "__segment_tiles"
    pairs = width == 32  # prefix2: 64-row tiles then 128-row pairs per (segment, head), warp-scanned
    define(prefix, ((2 if pairs else 1) * (groups+1),), None, "int64")
    work = outputs[-1]
    expected_work = (groups, heads, 3 if fwd else 6)
    if layouts[work].shape != expected_work or dtypes[work] != "int64":
        raise LoweringError("segmented attention work counters have an invalid layout")
    if fwd:
        out, lse = outputs[:2]
        delta = None
        for name, expected, dtype in ((out, shape, "bfloat16"), (lse, (heads, rows), "float32")):
            if layouts[name].shape != expected or dtypes[name] != dtype:
                raise LoweringError("segmented attention output/statistics ABI differs")
    else:
        out, lse = ins[3:5]
        delta = outputs[0] + "__segment_delta"
        define(delta, (heads, rows), None, "float32")
        if any(layouts[n].shape != shape or dtypes[n] != "bfloat16" for n in outputs[:3]):
            raise LoweringError("segmented attention gradients must preserve packed BF16 shape")

    def ptr(n):
        l = layouts[n]
        return f"({ident(l.root)} + {l.offset}LL)"

    common = dict(threads=256, requires_full_grid=True, source_commit="P-OPS4-D v7 packed-segment adaptation",
                  arithmetic_contract=CONTRACT + ("+ftz-ex2" if variant == 6 else "+ftz-ex2+fused-bwd" if fused
                                                  else "+ftz-ex2+folded-fwd" if lean_forward else ""),
                  packed_rows=rows, segments=groups, heads=heads,
                  max_seqlen=bound, row_capacity_padding=0, head_width=width)
    phases = []

    def phase(kind, code, reads, writes, items, shared=0, **meta):
        phases.append(dict(kind=kind, names=list(writes), targets=[target], reads=list(reads), writes=list(writes),
                           code=code, work_items=max(1, items), shared_bytes=shared, **common, **meta))

    if fused:
        return _fused_backward(phase, ptr=ptr, q=q, k=k, v=v, out=out, lse=lse, dout=ins[5],
                               offsets=offsets, prefix=prefix, work=work, delta=delta, outputs=outputs, define=define,
                               rows=rows, heads=heads, groups=groups, bound=bound, scale=scale, phases=phases)
    phase("segmented_attention_prepare", f"""latch_segmented::{"prepare2" if pairs else "prepare"}(
  {ptr(offsets)}, {ptr(prefix)}, {ptr(work)}, {groups}, {heads}, {rows}, {bound}, {3 if fwd else 6});""",
        [offsets], [prefix, work], 1, work_counter_columns=list(FORWARD_WORK if fwd else BACKWARD_WORK))
    # Prefix is over real tile counts, not group*max_seqlen. It is recomputed from
    # device offsets every launch, so membership changes never change compilation.
    fields = f"""latch_segmented::Args A{{}};
A.q={ptr(q)}; A.k={ptr(k)}; A.v={ptr(v)}; A.o={ptr(out)};
A.lse2={ptr(lse)}; A.rows={rows}; A.heads={heads}; A.groups={groups};
A.scale={float(scale)!r}f; A.scale_mode=1; A.work={ptr(work)};
"""
    if not fwd:
        fields += f"A.dout={ptr(ins[5])}; A.delta={ptr(delta)}; A.dq={ptr(outputs[0])}; A.dk={ptr(outputs[1])}; A.dv={ptr(outputs[2])};\n"
    max_tiles = (rows + 63*groups)//64 * heads
    for which in (("forward",) if fwd else ("dq", "kv")):
        writes = [out, lse, work] if which == "forward" else [outputs[0], delta, work] if which == "dq" else [*outputs[1:3], work]
        reads = [q, k, v, offsets, prefix, work]
        if not fwd:
            reads += [out, lse, ins[5]]
            if which == "kv":
                reads += [delta, outputs[0]]
        # Work counter reads serialize prepare/dQ/dKV and keep their additions
        # alive even when the diagnostic result itself is not a model output.
        fx = FORWARD_FORM if variant == 8 and which == "forward" else 0
        code = fields + f"latch_segmented::run_{which}<{width},{variant + 16 * fx}>(A, {ptr(offsets)}, {ptr(prefix)}, smem);"
        phase("segmented_attention_"+which, code, reads, writes, (max_tiles+1)//2,
              1024 + 2 * 88 * 1024 if variant == 8 else 103424,
              work_counters=work, work_counter_columns=list(FORWARD_WORK if fwd else BACKWARD_WORK),
              native_body=body, dynamic_tiles=True)
    return phases


def _fused_backward(phase, *, ptr, q, k, v, out, lse, dout, offsets, prefix, work, delta, outputs, define,
                    rows, heads, groups, bound, scale, phases):
    dq, dk, dv = outputs[:3]
    dqacc = dq + "__fused_dqacc"
    define(dqacc, (rows, heads, 32), None, "float32")
    phase("segmented_attention_prepare", f"""latch_segmented::fused_bprep(
  {ptr(offsets)}, {ptr(prefix)}, {ptr(work)}, {groups}, {heads}, {rows}, {bound},
  {ptr(dout)}, {ptr(out)}, {ptr(delta)}, {ptr(dqacc)});""",
          [offsets, dout, out], [prefix, work, delta, dqacc], max(1, rows * heads // 256),
          work_counter_columns=list(BACKWARD_WORK))
    fields = f"""latch_segmented::Args A{{}};
A.q={ptr(q)}; A.k={ptr(k)}; A.v={ptr(v)}; A.o={ptr(out)};
A.lse2={ptr(lse)}; A.rows={rows}; A.heads={heads}; A.groups={groups};
A.scale={float(scale)!r}f; A.scale_mode=1; A.work={ptr(work)};
A.dout={ptr(dout)}; A.delta={ptr(delta)}; A.dq={ptr(dq)}; A.dk={ptr(dk)}; A.dv={ptr(dv)}; A.dqa={ptr(dqacc)};
"""
    max_tiles = (rows + 63 * groups) // 64 * heads
    phase("segmented_attention_kv", fields + f"latch_segmented::run_kv<32,{8 + 16 * BACKWARD_FORM}>(A, {ptr(offsets)}, {ptr(prefix)}, smem);",
          [q, k, v, offsets, prefix, work, lse, dout, delta, dqacc], [dk, dv, dqacc, work], (max_tiles + 1) // 2, 1024 + 2 * 88 * 1024,
          work_counters=work, work_counter_columns=list(BACKWARD_WORK), native_body=native_body(32, 8),
          dynamic_tiles=True)
    convert = f"latch_segmented::fused_dqcvt_hr({ptr(dqacc)}, {ptr(dq)}, {rows}, {heads}, {float(scale)!r}f);"
    phase("segmented_attention_dq", convert, [dqacc, work], [dq], max(1, rows * heads // 64),
          work_counters=work, work_counter_columns=list(BACKWARD_WORK))
    return phases
